import json
import os
from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.models import EventSchema

# Get Ollama URL from environment variable (for Docker support)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# 1. Define the State
class AgentState(TypedDict):
    messages: list  # Chat history
    event_draft: Dict[str, Any]  # Current state of the JSON object
    schema_definition: str  # The API requirement description
    next_step: str  # Internal flag for the agent flow

# 2. Setup LLM
# Ensure you have Ollama running: `ollama run llama3` (or mistral, etc.)
llm = ChatOllama(
    model="gpt-oss:20b", 
    temperature=0, 
    format="json",
    base_url=OLLAMA_URL
)
conversational_llm = ChatOllama(
    model="gpt-oss:20b", 
    temperature=0.7,
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
    """

    response = llm.invoke([SystemMessage(content=system_prompt)])
    
    try:
        # Llama3 in JSON mode usually returns clean JSON, but we ensure parsing
        updates = json.loads(response.content)
    except json.JSONDecodeError:
        updates = {}

    # Merge updates into draft
    updated_draft = current_draft.copy()
    updated_draft.update(updates)

    return {"event_draft": updated_draft}

# 5. Node: Determine Status & Formulate Response
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

# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("extract_data", data_extractor)
workflow.add_node("generate_response", response_generator)

workflow.set_entry_point("extract_data")
workflow.add_edge("extract_data", "generate_response")
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
        "next_step": ""
    }
    
    result = await app_graph.ainvoke(initial_state)
    
    last_ai_message = result["messages"][-1].content
    updated_draft = result["event_draft"]
    
    return last_ai_message, updated_draft