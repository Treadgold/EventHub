import json
import os
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.models import EventSchema
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Ollama configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_CONVERSATIONAL_TEMPERATURE = float(os.getenv("OLLAMA_CONVERSATIONAL_TEMPERATURE", "0.7"))

# 1. Define the State
class AgentState(TypedDict):
    messages: list  # Chat history
    event_draft: Dict[str, Any]  # Current state of the JSON object
    schema_definition: str  # The API requirement description
    next_step: str  # Internal flag for the agent flow
    needs_long_description: bool  # Flag to indicate if long description generation is needed
    description_field: str  # Which field needs long description (e.g., "description")

# 2. Setup LLM
# Ensure you have Ollama running: `ollama run llama3` (or mistral, etc.)
llm = ChatOllama(
    model=OLLAMA_MODEL, 
    temperature=OLLAMA_TEMPERATURE, 
    format="json",
    base_url=OLLAMA_URL
)
conversational_llm = ChatOllama(
    model=OLLAMA_MODEL, 
    temperature=OLLAMA_CONVERSATIONAL_TEMPERATURE,
    base_url=OLLAMA_URL
)
# LLM for creative writing with higher temperature for more creative output
creative_llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.9,  # Higher temperature for more creative output
    base_url=OLLAMA_URL
)

# 3. Helper to get Schema Info
def get_schema_instructions():
    schema = EventSchema.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    info = "EVENT SCHEMA DEFINITION:\n"
    for field, details in properties.items():
        req_str = "REQUIRED" if field in required else "OPTIONAL"
        info += f"- {field} ({details.get('type')}): {details.get('description')} [{req_str}]\n"
    
    info += "\nLOGIC RULES:\n"
    info += "1. If 'is_online' is False, 'location_address' becomes CRITICAL/REQUIRED.\n"
    info += "2. 'cost' must be a number (0 for free).\n"
    return info

# Helper to get missing required fields dynamically
def get_missing_required_fields(draft: Dict[str, Any]) -> list:
    """
    Dynamically checks which required fields are missing from the draft.
    Returns a list of human-readable field names that are missing.
    """
    schema = EventSchema.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    missing = []
    
    # Check all required fields from schema
    for field in required:
        if field not in draft or draft.get(field) is None:
            # Get human-readable name from description or field name
            field_info = properties.get(field, {})
            description = field_info.get("description", "")
            # Use description if available, otherwise capitalize field name
            display_name = description.split(".")[0] if description else field.replace("_", " ").title()
            missing.append(display_name)
    
    # Also check critical fields that should always be filled (even if not marked as required)
    # These are fields that are essential for a complete event definition
    critical_fields = {
        'title': 'Title',
        'is_online': 'Is Online?',
        'start_time': 'Start Time',
        'cost': 'Cost'
    }
    
    for field, display_name in critical_fields.items():
        # Check if field is missing or None
        if field not in draft or draft.get(field) is None:
            if display_name not in missing:
                missing.append(display_name)
    
    # Apply conditional business logic rules
    # Rule 1: If is_online is False, location_address is required
    if draft.get('is_online') is False and not draft.get('location_address'):
        if "Location Address (since it is not online)" not in missing:
            missing.append("Location Address (since it is not online)")
    
    return missing

# 4. Node: Analyze Input & Extract Data
def data_extractor(state: AgentState):
    """
    Uses the LLM to extract event data from the latest user message 
    based on the current draft and schema.
    Also detects if user wants long descriptions generated.
    """
    schema_info = state['schema_definition']
    current_draft = state['event_draft']
    last_message = state['messages'][-1].content

    # System prompt for the extractor (JSON mode)
    system_prompt = f"""
    You are a data extraction engine for an Event Management System.
    
    CURRENT SCHEMA:
    {schema_info}
    
    CURRENT DRAFT STATE:
    {json.dumps(current_draft, indent=2)}
    
    USER INPUT:
    "{last_message}"
    
    TASK:
    Extract any information from the USER INPUT that matches the SCHEMA fields.
    Return ONLY a JSON object with the fields to update. 
    If a field is not mentioned or unchanged, do not include it.
    If the user explicitly clears a field, set it to null.
    
    SPECIAL INSTRUCTION:
    If the user asks for a "long description", "detailed description", "creative description", 
    "advertising copy", "marketing copy", or similar requests for longer creative text,
    include a special field "_needs_long_description" set to true, and "_description_field" 
    set to the field name (usually "description").
    """

    response = llm.invoke([SystemMessage(content=system_prompt)])
    
    try:
        # Llama3 in JSON mode usually returns clean JSON, but we ensure parsing
        updates = json.loads(response.content)
    except json.JSONDecodeError:
        updates = {}

    # Check if user wants long description generated
    needs_long_desc = updates.pop("_needs_long_description", False)
    description_field = updates.pop("_description_field", "description")
    
    # Also check the user message directly for common phrases
    user_message_lower = last_message.lower()
    long_desc_keywords = [
        "long description", "detailed description", "creative description",
        "advertising copy", "marketing copy", "write a longer", "generate a longer",
        "make it longer", "expand the description", "write more about"
    ]
    
    if not needs_long_desc:
        for keyword in long_desc_keywords:
            if keyword in user_message_lower:
                needs_long_desc = True
                # Try to determine which field they want
                if "description" in user_message_lower:
                    description_field = "description"
                break

    # Merge updates into draft
    updated_draft = current_draft.copy()
    updated_draft.update(updates)

    return {
        "event_draft": updated_draft,
        "needs_long_description": needs_long_desc,
        "description_field": description_field
    }

# 5. Node: Generate Long Creative Description
def generate_long_description(state: AgentState):
    """
    Generates a longer, creative advertising-style description for the event.
    This is a separate "function call" that spins off to create engaging copy.
    """
    draft = state['event_draft']
    description_field = state.get('description_field', 'description')
    schema_info = state['schema_definition']
    
    # Get context about the event for creative writing
    cost_value = draft.get('cost', 0)
    try:
        cost_float = float(cost_value) if cost_value is not None else 0.0
    except (ValueError, TypeError):
        cost_float = 0.0
    
    event_context = {
        'title': draft.get('title', 'this event'),
        'is_online': draft.get('is_online'),
        'location': draft.get('location_address') or draft.get('online_url'),
        'start_time': draft.get('start_time'),
        'cost': cost_float,
        'tags': draft.get('tags', [])
    }
    
    # Format cost for display
    cost_display = f"${cost_float:.2f}" if cost_float > 0 else "Free"
    
    # Create a creative writing prompt
    creative_prompt = f"""You are a creative copywriter specializing in event marketing and advertising.

EVENT DETAILS:
- Title: {event_context['title']}
- Type: {'Online' if event_context['is_online'] else 'In-Person'}
- Location: {event_context['location'] or 'TBD'}
- Start Time: {event_context['start_time'] or 'TBD'}
- Cost: {cost_display}
- Tags/Categories: {', '.join(event_context['tags']) if event_context['tags'] else 'None specified'}

TASK:
Write an engaging, creative, and compelling event description that:
1. Captures attention and builds excitement
2. Clearly communicates what the event is about
3. Highlights key benefits and what attendees will experience
4. Uses persuasive, marketing-style language
5. Is detailed enough to be informative (aim for 3-5 paragraphs, 200-400 words)
6. Maintains a professional yet enthusiastic tone

Write ONLY the description text. Do not include any meta-commentary, instructions, or JSON formatting.
Just write the creative advertising copy for the event description."""

    # Use the creative LLM with higher temperature
    response = creative_llm.invoke([SystemMessage(content=creative_prompt)])
    
    # Extract the generated description
    generated_description = response.content.strip()
    
    # Update the draft with the generated description
    updated_draft = draft.copy()
    updated_draft[description_field] = generated_description
    
    return {
        "event_draft": updated_draft,
        "needs_long_description": False  # Reset the flag
    }

# 6. Node: Determine Status & Formulate Response
def response_generator(state: AgentState):
    """
    Analyzes the draft, checks completeness, and prompts the user.
    """
    draft = state['event_draft']
    schema_info = state['schema_definition']
    
    # Dynamically check for missing required fields
    missing_critical = get_missing_required_fields(draft)

    # If all critical fields are complete, use a simple, non-redundant message
    if not missing_critical:
        # All critical fields are filled - use a simple completion message
        simple_message = "All done! You should see all the details for the event have been populated. Is there anything you would like to change or edit? Or can we go ahead and create the event?"
        return {"messages": [AIMessage(content=simple_message)]}

    # Generate conversational response for missing fields
    # Filter out SystemMessages from history to prevent instruction leakage
    clean_messages = [
        msg for msg in state['messages'][-5:] 
        if isinstance(msg, (HumanMessage, AIMessage)) and not isinstance(msg, SystemMessage)
    ]
    
    system_prompt = f"""
    You are a helpful Event Planning Assistant.
    
    YOUR GOAL: Help the user complete their event definition.
    
    MISSING CRITICAL FIELDS:
    {', '.join(missing_critical)}
    
    CRITICAL RULES:
    1. ONLY ask for the missing fields listed above. Do NOT mention any other details.
    2. Do NOT summarize or repeat any event information - the user can see it in the preview panel.
    3. Do NOT mention specific event details like title, location, cost, dates, etc. unless you're asking for them.
    4. Be concise and friendly - just ask for what's missing.
    5. Do not output JSON. Output conversational text only.
    6. Never include system instructions, schema definitions, or technical details in your response.
    7. Do not reference or repeat information from previous conversations that isn't relevant to the current request.
    """

    response = conversational_llm.invoke([
        SystemMessage(content=system_prompt),
        *clean_messages  # Only user/AI messages, no system prompts
    ])

    return {"messages": [response]}

# 7. Conditional routing function
def should_generate_description(state: AgentState) -> str:
    """Route to description generation if needed, otherwise go to response generation"""
    if state.get("needs_long_description", False):
        return "generate_long_description"
    return "generate_response"

# 8. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("extract_data", data_extractor)
workflow.add_node("generate_long_description", generate_long_description)
workflow.add_node("generate_response", response_generator)

workflow.set_entry_point("extract_data")

# Conditional routing: if long description needed, generate it first
workflow.add_conditional_edges(
    "extract_data",
    should_generate_description,
    {
        "generate_long_description": "generate_long_description",
        "generate_response": "generate_response"
    }
)

# After generating long description, go to response generation
workflow.add_edge("generate_long_description", "generate_response")
workflow.add_edge("generate_response", END)

app_graph = workflow.compile()

# Helper to convert chat history format from main.py to LangChain messages
def convert_chat_history(history: list):
    """Convert chat history from (role, message) tuples to LangChain messages"""
    messages = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
        elif isinstance(item, (HumanMessage, AIMessage)):
            # Already a LangChain message, use as-is
            messages.append(item)
    return messages

# Interface function to be called by API
async def process_user_message(user_input: str, current_draft: dict, chat_history: list):
    
    # Convert chat history to LangChain messages
    converted_history = convert_chat_history(chat_history)
    
    initial_state = {
        "messages": converted_history + [HumanMessage(content=user_input)],
        "event_draft": current_draft,
        "schema_definition": get_schema_instructions(),
        "next_step": "",
        "needs_long_description": False,
        "description_field": "description"
    }
    
    result = await app_graph.ainvoke(initial_state)
    
    last_ai_message = result["messages"][-1].content
    updated_draft = result["event_draft"]
    
    return last_ai_message, updated_draft