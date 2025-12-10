from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from app.agent import process_user_message
from app.models import EventSchema
from app.database import connect_to_mongo, close_mongo_connection, get_database
from app.db_models import User, Event, event_indexes, user_indexes
from app.auth import (
    get_password_hash, verify_password, get_current_user, create_access_token,
    require_authenticated, require_admin, require_event_creator, require_roles,
    can_create_events, can_edit_event, can_delete_event, can_edit_users,
    UserRole
)
from app.whisper import check_whisper_status, transcribe_audio, WhisperStatus
from bson import ObjectId
from datetime import datetime
import json
import dateutil.parser
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EventHub")
templates = Jinja2Templates(directory="app/templates")

# Session middleware with proper cookie configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
SESSION_MAX_AGE_DAYS = int(os.getenv("SESSION_MAX_AGE_DAYS", "30"))
SESSION_SAME_SITE = os.getenv("SESSION_SAME_SITE", "lax")
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() == "true"

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session",
    max_age=60 * 60 * 24 * SESSION_MAX_AGE_DAYS,  # From environment variable
    same_site=SESSION_SAME_SITE,
    https_only=SESSION_HTTPS_ONLY,
)

# Custom Jinja2 filter for human-readable date formatting
def format_datetime(value):
    """Convert ISO datetime string or datetime object to human-readable format"""
    if not value:
        return value
    
    # If it's already a datetime object
    if isinstance(value, datetime):
        return value.strftime('%A, %B %d, %Y at %I:%M %p')
    
    # If it's a string, try to parse it
    if isinstance(value, str):
        if value == '...':
            return value
        
        # Try to parse ISO format first
        for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime('%A, %B %d, %Y at %I:%M %p')
            except (ValueError, TypeError):
                continue
        
        # Try dateutil parser for natural language dates
        try:
            dt = dateutil.parser.parse(value)
            return dt.strftime('%A, %B %d, %Y at %I:%M %p')
        except:
            pass
    
    # If parsing fails, return original
    return value

templates.env.filters['format_datetime'] = format_datetime


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

async def create_test_users():
    """Create test users for each role category"""
    db = get_database()
    
    test_users = [
        # Admins
        {
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123",
            "role": UserRole.ADMIN
        },
        {
            "username": "admin2",
            "email": "admin2@example.com",
            "password": "admin123",
            "role": UserRole.ADMIN
        },
        # Event Organisers
        {
            "username": "organiser1",
            "email": "organiser1@example.com",
            "password": "organiser123",
            "role": UserRole.EVENT_ORGANISER
        },
        {
            "username": "organiser2",
            "email": "organiser2@example.com",
            "password": "organiser123",
            "role": UserRole.EVENT_ORGANISER
        },
        # Regular Users
        {
            "username": "user1",
            "email": "user1@example.com",
            "password": "user123",
            "role": UserRole.USER
        },
        {
            "username": "user2",
            "email": "user2@example.com",
            "password": "user123",
            "role": UserRole.USER
        },
    ]
    
    print("\n" + "="*60)
    print("Creating/Updating Test Users")
    print("="*60)
    
    for user_data in test_users:
        try:
            existing_user = await db.users.find_one({"username": user_data["username"]})
            
            if existing_user:
                # Update existing user to ensure correct role
                if existing_user.get("role") != user_data["role"]:
                    await db.users.update_one(
                        {"username": user_data["username"]},
                        {"$set": {"role": user_data["role"]}}
                    )
                    print(f"  ✓ Updated '{user_data['username']}' to role: {user_data['role']}")
                else:
                    print(f"  ✓ User '{user_data['username']}' exists with role: {user_data['role']}")
            else:
                # Create new user
                password_hash = get_password_hash(str(user_data["password"]))
                new_user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    password_hash=password_hash,
                    role=user_data["role"]
                )
                await db.users.insert_one(new_user.dict(by_alias=True))
                print(f"  ✓ Created '{user_data['username']}' with role: {user_data['role']}")
        except Exception as e:
            print(f"  ✗ Error with user '{user_data['username']}': {e}")
    
    print("\n" + "-"*60)
    print("Test User Credentials:")
    print("-"*60)
    print("  ADMINS (can create, edit, delete events & manage users):")
    print("    - admin / admin123")
    print("    - admin2 / admin123")
    print("\n  EVENT ORGANISERS (can create & edit own events):")
    print("    - organiser1 / organiser123")
    print("    - organiser2 / organiser123")
    print("\n  REGULAR USERS (can view events & buy tickets):")
    print("    - user1 / user123")
    print("    - user2 / user123")
    print("="*60 + "\n")


@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    db = get_database()
    # Create indexes
    await db.events.create_indexes(event_indexes)
    await db.users.create_indexes(user_indexes)
    # Create test users
    await create_test_users()


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_template_user(request: Request):
    """Get current user for templates"""
    return await get_current_user(request)


# ============================================================================
# PUBLIC ROUTES (no authentication required)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - shows events in chronological order (public)"""
    db = get_database()
    current_user = await get_template_user(request)
    
    # Get all upcoming events, sorted by start_time (chronological)
    cursor = db.events.find({"start_time": {"$gte": datetime.utcnow()}}).sort("start_time", 1)
    events_list = await cursor.to_list(length=100)
    
    # Convert to Event models
    events = []
    for event_doc in events_list:
        event = Event(**event_doc)
        events.append(event)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "events": events,
        "current_user": current_user
    })


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page (public)"""
    current_user = await get_template_user(request)
    return templates.TemplateResponse("about.html", {
        "request": request,
        "current_user": current_user
    })


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    current_user = await get_template_user(request)
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "current_user": None,
        "error": None
    })


@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process login form"""
    logger.info(f"Login attempt for username: {username}")
    db = get_database()
    
    # Find user
    user_doc = await db.users.find_one({"username": username})
    if not user_doc:
        logger.warning(f"Login failed: User '{username}' not found")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "current_user": None,
            "error": "Invalid username or password"
        })
    
    # Ensure role field exists (for users created before role was added)
    if "role" not in user_doc:
        if user_doc.get("username") == "admin":
            user_doc["role"] = UserRole.ADMIN
        else:
            user_doc["role"] = UserRole.USER
        # Update the database
        await db.users.update_one(
            {"_id": user_doc["_id"]},
            {"$set": {"role": user_doc["role"]}}
        )
    
    user = User(**user_doc)
    
    # Verify password
    if not verify_password(password, user.password_hash):
        logger.warning(f"Login failed: Invalid password for user '{username}'")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "current_user": None,
            "error": "Invalid username or password"
        })
    
    # Set session - use the actual MongoDB _id from the document, not the model's id
    # (the model might generate a new ObjectId due to default_factory)
    actual_user_id = str(user_doc["_id"])
    request.session["user_id"] = actual_user_id
    logger.info(f"Login successful: User '{username}' (role: {user.role}) - session user_id set to {actual_user_id}")
    
    # Redirect based on role - use 303 See Other for POST redirect (proper HTTP semantics)
    if user.role in [UserRole.ADMIN, UserRole.EVENT_ORGANISER]:
        redirect_url = "/create-event"
    else:
        redirect_url = "/"
    
    logger.info(f"Redirecting user '{username}' to {redirect_url}")
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    current_user = await get_template_user(request)
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {
        "request": request,
        "current_user": None,
        "error": None
    })


@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """Process registration form"""
    db = get_database()
    
    # Check if username or email already exists
    existing_user = await db.users.find_one({
        "$or": [{"username": username}, {"email": email}]
    })
    if existing_user:
        return templates.TemplateResponse("register.html", {
            "request": request,
            "current_user": None,
            "error": "Username or email already exists"
        })
    
    # Create new user (default role is "user" - they can view events and buy tickets)
    password_hash = get_password_hash(password)
    new_user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=UserRole.USER
    )
    
    result = await db.users.insert_one(new_user.dict(by_alias=True))
    
    # Set session with the actual inserted ID
    request.session["user_id"] = str(result.inserted_id)
    logger.info(f"Registration successful: User '{username}' - session user_id set to {result.inserted_id}")
    
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


# ============================================================================
# EVENT CREATION ROUTES (require admin or event_organiser role)
# ============================================================================

@app.get("/create-event", response_class=HTMLResponse)
async def create_event_page(request: Request):
    """Event creation page (requires admin or event_organiser role)"""
    logger.info(f"GET /create-event - Session: {dict(request.session)}")
    current_user = await get_template_user(request)
    
    # Check if user is logged in
    if not current_user:
        logger.warning("GET /create-event - No authenticated user, redirecting to login")
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Check if user has permission to create events
    if not can_create_events(current_user):
        return templates.TemplateResponse("error.html", {
            "request": request,
            "current_user": current_user,
            "error": "You don't have permission to create events. Only admins and event organisers can create events."
        }, status_code=403)
    
    # Initialize session for event creation
    session_id = f"event_draft_{current_user.id}"
    if session_id not in request.session:
        request.session[session_id] = {
            "draft": {},
            "history": []
        }
    
    # Get schema metadata for dynamic rendering
    schema = EventSchema.model_json_schema()
    session_data = request.session[session_id]
    draft = session_data.get("draft", {})
    history = session_data.get("history", [])
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "draft": draft,
        "history": history,
        "schema": schema,
        "current_user": current_user
    })


@app.post("/chat", response_class=HTMLResponse)
async def chat_endpoint(
    request: Request,
    user_message: str = Form(...)
):
    """Chat endpoint for event creation AI (requires admin or event_organiser role)"""
    current_user = await get_template_user(request)
    
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if not can_create_events(current_user):
        raise HTTPException(status_code=403, detail="You don't have permission to create events")
    
    session_id = f"event_draft_{current_user.id}"
    session_data = request.session.get(session_id, {"draft": {}, "history": []})
    
    # Call Agent
    response_text, updated_draft = await process_user_message(
        user_message,
        session_data.get("draft", {}),
        session_data.get("history", [])
    )
    
    # Update Session
    session_data["draft"] = updated_draft
    session_data["history"].append(("user", user_message))
    session_data["history"].append(("ai", response_text))
    request.session[session_id] = session_data
    
    # Get schema metadata for dynamic rendering
    schema = EventSchema.model_json_schema()
    
    return templates.TemplateResponse("components/chat_response.html", {
        "request": request,
        "user_message": user_message,
        "ai_message": response_text,
        "draft": updated_draft,
        "schema": schema,
        "current_user": current_user
    })


@app.post("/save-event", response_class=HTMLResponse)
async def save_event(request: Request):
    """Save the drafted event to the database (requires admin or event_organiser role)"""
    current_user = await get_template_user(request)
    
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if not can_create_events(current_user):
        raise HTTPException(status_code=403, detail="You don't have permission to create events")
    
    session_id = f"event_draft_{current_user.id}"
    session_data = request.session.get(session_id, {})
    draft = session_data.get("draft", {}).copy()  # Work with a copy to avoid modifying session
    
    # Validate draft using EventSchema
    try:
        # Parse start_time and end_time if they're strings
        parsed_start_time = None
        parsed_end_time = None
        
        if "start_time" in draft and draft["start_time"]:
            if isinstance(draft["start_time"], str):
                try:
                    parsed_start_time = dateutil.parser.parse(draft["start_time"])
                except:
                    raise ValueError("Invalid start_time format")
            else:
                parsed_start_time = draft["start_time"]
        
        if "end_time" in draft and draft["end_time"]:
            if isinstance(draft["end_time"], str):
                try:
                    parsed_end_time = dateutil.parser.parse(draft["end_time"])
                except:
                    pass  # end_time is optional
            else:
                parsed_end_time = draft["end_time"]
        
        # Validate required fields
        if not draft.get("title"):
            raise ValueError("Title is required")
        if draft.get("is_online") is None:
            raise ValueError("is_online is required")
        if not parsed_start_time:
            raise ValueError("start_time is required")
        if draft.get("is_online") is False and not draft.get("location_address"):
            raise ValueError("location_address is required for in-person events")
        
        # Create Event object
        event = Event(
            title=draft["title"],
            description=draft.get("description"),
            is_online=draft["is_online"],
            location_address=draft.get("location_address"),
            online_url=draft.get("online_url"),
            start_time=parsed_start_time,
            end_time=parsed_end_time,
            cost=draft.get("cost", 0.0),
            tags=draft.get("tags", []),
            media_urls=draft.get("media_urls", []),
            created_by=str(current_user.id)  # Pass as string, PyObjectId will validate
        )
        
        # Save to MongoDB - exclude _id to let MongoDB generate a proper ObjectId
        db = get_database()
        event_data = event.model_dump(by_alias=True, exclude={'id'})
        result = await db.events.insert_one(event_data)
        
        # Clear session
        request.session[session_id] = {"draft": {}, "history": []}
        
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        
    except Exception as e:
        # Return error message (draft is still in string format, safe for templates)
        schema = EventSchema.model_json_schema()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "draft": draft,
            "schema": schema,
            "current_user": current_user,
            "error": str(e)
        })


@app.post("/clear-draft", response_class=HTMLResponse)
async def clear_draft(request: Request):
    """Clear the event draft from session (requires admin or event_organiser role)"""
    current_user = await get_template_user(request)
    
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if not can_create_events(current_user):
        raise HTTPException(status_code=403, detail="You don't have permission to create events")
    
    session_id = f"event_draft_{current_user.id}"
    
    # Clear the draft and history from session
    request.session[session_id] = {"draft": {}, "history": []}
    
    # Redirect back to create-event page
    return RedirectResponse(url="/create-event", status_code=status.HTTP_302_FOUND)


# ============================================================================
# ADMIN ROUTES (require admin role)
# ============================================================================

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    current_user: User = Depends(require_admin)
):
    """Admin page to manage users"""
    db = get_database()
    
    cursor = db.users.find().sort("created_at", -1)
    users_list = await cursor.to_list(length=100)
    users = [User(**u) for u in users_list]
    
    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "users": users,
        "current_user": current_user,
        "UserRole": UserRole
    })


@app.post("/admin/users/{user_id}/role", response_class=HTMLResponse)
async def update_user_role(
    request: Request,
    user_id: str,
    role: str = Form(...),
    current_user: User = Depends(require_admin)
):
    """Update a user's role (admin only)"""
    db = get_database()
    
    # Validate role
    if role not in [UserRole.ADMIN, UserRole.EVENT_ORGANISER, UserRole.USER]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    # Don't allow admin to remove their own admin role
    if str(current_user.id) == user_id and role != UserRole.ADMIN:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin role")
    
    # Update user role
    result = await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"role": role}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return RedirectResponse(url="/admin/users", status_code=status.HTTP_302_FOUND)


@app.post("/admin/users/{user_id}/delete")
async def delete_user(
    request: Request,
    user_id: str,
    current_user: User = Depends(require_admin)
):
    """Delete a user (admin only)"""
    db = get_database()
    
    # Don't allow admin to delete themselves
    if str(current_user.id) == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    result = await db.users.delete_one({"_id": ObjectId(user_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return RedirectResponse(url="/admin/users", status_code=status.HTTP_302_FOUND)


# ============================================================================
# WHISPER SPEECH-TO-TEXT ROUTES
# ============================================================================

@app.get("/whisper/config", response_class=HTMLResponse)
async def whisper_config_page(request: Request):
    """Whisper configuration page"""
    current_user = await get_template_user(request)
    whisper_status = await check_whisper_status()
    
    return templates.TemplateResponse("whisper_config.html", {
        "request": request,
        "current_user": current_user,
        "whisper_status": whisper_status
    })


@app.get("/whisper/status", response_class=HTMLResponse)
async def whisper_status_endpoint(request: Request):
    """Get Whisper server status (for HTMX refresh)"""
    whisper_status = await check_whisper_status()
    
    return templates.TemplateResponse("components/whisper_status.html", {
        "request": request,
        "whisper_status": whisper_status
    })


@app.get("/whisper/available")
async def whisper_available():
    """Check if Whisper is available (JSON API for frontend)"""
    whisper_status = await check_whisper_status()
    return {
        "available": whisper_status.status == WhisperStatus.AVAILABLE,
        "status": whisper_status.status,
        "message": whisper_status.message
    }


@app.post("/whisper/transcribe")
async def whisper_transcribe(request: Request):
    """Transcribe audio file"""
    from fastapi import UploadFile, File
    
    # Get form data with file
    form = await request.form()
    audio_file = form.get("audio")
    
    if not audio_file:
        return {"success": False, "error": "No audio file provided"}
    
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        if len(audio_data) == 0:
            return {"success": False, "error": "Empty audio file"}
        
        # Transcribe
        result = await transcribe_audio(
            audio_data=audio_data,
            filename=getattr(audio_file, 'filename', 'audio.wav')
        )
        
        return {
            "success": result.success,
            "text": result.text,
            "error": result.error,
            "language": result.language
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"success": False, "error": str(e)}
