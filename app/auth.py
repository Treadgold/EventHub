from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List, Callable
import os
import logging
from app.db_models import User
from app.database import get_database
from bson import ObjectId

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

security = HTTPBearer()

# User roles
class UserRole:
    ADMIN = "admin"
    EVENT_ORGANISER = "event_organiser"
    USER = "user"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(request: Request) -> Optional[User]:
    """Get current user from session cookie or JWT token"""
    # Try to get from session cookie first
    user_id = request.session.get("user_id")
    logger.debug(f"Session check - user_id from session: {user_id}")
    
    if user_id:
        try:
            db = get_database()
            # Try to find user - handle both ObjectId and string _id formats
            user_doc = None
            try:
                # First try as ObjectId
                user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
            except:
                pass
            
            if not user_doc:
                # Try as string (in case _id was stored as string)
                user_doc = await db.users.find_one({"_id": user_id})
            
            if user_doc:
                # Ensure role field exists (for users created before role was added)
                if "role" not in user_doc:
                    user_doc["role"] = UserRole.USER
                logger.info(f"User authenticated via session: {user_doc.get('username')} (role: {user_doc.get('role')})")
                return User(**user_doc)
            else:
                # User ID in session but user doesn't exist in database
                # This can happen if database was reset - clear the invalid session
                logger.warning(f"Session has user_id {user_id} but user not found in database - clearing session")
                request.session.clear()
        except Exception as e:
            logger.error(f"Error looking up user from session: {e}")
            request.session.clear()
    
    # Try to get from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if user_id:
                db = get_database()
                user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
                if user_doc:
                    # Ensure role field exists (for users created before role was added)
                    if "role" not in user_doc:
                        user_doc["role"] = UserRole.USER
                    logger.info(f"User authenticated via JWT: {user_doc.get('username')}")
                    return User(**user_doc)
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
    
    logger.debug("No authenticated user found")
    return None


# ============================================================================
# DEPENDENCY INJECTION FUNCTIONS (use with FastAPI's Depends)
# ============================================================================

async def require_authenticated(request: Request) -> User:
    """
    Dependency: Require user to be authenticated.
    Use: current_user: User = Depends(require_authenticated)
    """
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(request: Request) -> User:
    """
    Dependency: Require user to be an admin.
    Use: current_user: User = Depends(require_admin)
    """
    user = await require_authenticated(request)
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required."
        )
    return user


async def require_event_organiser(request: Request) -> User:
    """
    Dependency: Require user to be an event organiser.
    Use: current_user: User = Depends(require_event_organiser)
    """
    user = await require_authenticated(request)
    if user.role != UserRole.EVENT_ORGANISER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Event organiser access required."
        )
    return user


async def require_event_creator(request: Request) -> User:
    """
    Dependency: Require user to be admin or event organiser (can create events).
    Use: current_user: User = Depends(require_event_creator)
    """
    user = await require_authenticated(request)
    if user.role not in [UserRole.ADMIN, UserRole.EVENT_ORGANISER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and event organisers can create events."
        )
    return user


def require_roles(allowed_roles: List[str]):
    """
    Factory function to create a dependency that requires specific roles.
    Use: current_user: User = Depends(require_roles([UserRole.ADMIN, UserRole.EVENT_ORGANISER]))
    """
    async def role_checker(request: Request) -> User:
        user = await require_authenticated(request)
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        return user
    return role_checker


# ============================================================================
# PERMISSION HELPER FUNCTIONS
# ============================================================================

def can_create_events(user: User) -> bool:
    """Check if user can create events"""
    return user.role in [UserRole.ADMIN, UserRole.EVENT_ORGANISER]


def can_edit_event(user: User, event) -> bool:
    """Check if user can edit an event"""
    if user.role == UserRole.ADMIN:
        return True
    if user.role == UserRole.EVENT_ORGANISER:
        return str(event.created_by) == str(user.id)
    return False


def can_delete_event(user: User, event) -> bool:
    """
    Check if user can delete an event.
    - Admin: can delete any event
    - Event organiser: can only delete their own events if not live yet
    """
    if user.role == UserRole.ADMIN:
        return True
    if user.role == UserRole.EVENT_ORGANISER:
        # Event organisers can only delete their own events if they're not live yet
        if str(event.created_by) == str(user.id):
            # Event is "live" if start_time has passed
            return event.start_time > datetime.utcnow()
    return False


def can_edit_users(user: User) -> bool:
    """Check if user can edit other user accounts"""
    return user.role == UserRole.ADMIN


def can_buy_tickets(user: Optional[User]) -> bool:
    """Check if user can buy tickets (must be logged in)"""
    return user is not None


def is_admin(user: Optional[User]) -> bool:
    """Check if user is an admin"""
    return user is not None and user.role == UserRole.ADMIN


def is_event_organiser(user: Optional[User]) -> bool:
    """Check if user is an event organiser"""
    return user is not None and user.role == UserRole.EVENT_ORGANISER


# Legacy alias for backwards compatibility
async def get_current_user_required(request: Request) -> User:
    """Alias for require_authenticated"""
    return await require_authenticated(request)


async def require_role(request: Request, allowed_roles: list) -> User:
    """Legacy function - prefer using require_roles() factory or specific role dependencies"""
    user = await require_authenticated(request)
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
        )
    return user
