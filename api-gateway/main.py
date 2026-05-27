"""
API Gateway for Local LLM Integration with Home Assistant

This service routes requests to appropriate handlers:
- Home Assistant commands
- Web search queries
- Automation variation generation
"""

import os
import sys
import json
import logging
import re
import time
import uuid
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import httpx

# Add parent and current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir))

from ha_executor import (
    handle_ha_command as execute_ha_command,
    start_entity_cache_refresher,
    stop_entity_cache_refresher,
)

# Import from hyphenated directory names using importlib
import importlib.util

# Import ha-client (hyphenated directory name)
ha_client_spec = importlib.util.spec_from_file_location(
    "ha_client",
    Path(__file__).parent.parent / "ha-client" / "client.py"
)
ha_client_module = importlib.util.module_from_spec(ha_client_spec)
ha_client_spec.loader.exec_module(ha_client_module)
HomeAssistantClient = ha_client_module.HomeAssistantClient

# Import search-service (hyphenated directory name)
search_spec = importlib.util.spec_from_file_location(
    "search_service", 
    Path(__file__).parent.parent / "search-service" / "search.py"
)
search_module = importlib.util.module_from_spec(search_spec)
search_spec.loader.exec_module(search_module)
get_search_service = search_module.get_search_service

# Import automation-variations (hyphenated directory name)
automation_spec = importlib.util.spec_from_file_location(
    "automation_variations",
    Path(__file__).parent.parent / "automation-variations" / "generator.py"
)
automation_module = importlib.util.module_from_spec(automation_spec)
automation_spec.loader.exec_module(automation_module)
gen_variations = automation_module.generate_variations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _minutes_to_hhmmss(minutes: int) -> str:
    """Convert minutes to HH:MM:SS format for Home Assistant timer scripts."""
    total_seconds = minutes * 60
    hours = total_seconds // 3600
    minutes_remaining = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes_remaining:02d}:{seconds:02d}"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    ha_url: str
    ha_access_token: str
    vllm_url: str = "http://vllm:8000"
    vllm_api_key: str = "local-dev-key"
    api_port: int = 8080
    api_host: str = "0.0.0.0"
    log_level: str = "INFO"
    search_provider: str = "duckduckgo"  # Optional, for future use
    kb_db_path: str = "/opt/llm/knowledge-base/entities.db"  # Optional, for future use
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


settings = Settings()


# OpenAI-compatible Request/Response Models
class ChatMessage(BaseModel):
    """OpenAI message format"""
    role: str  # "system", "user", "assistant", "tool", "function"
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[str] = None
    # Custom extension: use_case hint (optional, for internal routing)
    use_case: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """OpenAI chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str  # "stop", "length", "tool_calls", "function_call", "content_filter"


class Usage(BaseModel):
    """OpenAI token usage"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class VariationRequest(BaseModel):
    """Request model for automation variation generation"""
    base_message: str
    count: Optional[int] = 1


class VariationResponse(BaseModel):
    """Response model for automation variations"""
    variations: list[str]


# Global clients
ha_client: Optional[HomeAssistantClient] = None

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks"""
    global ha_client
    logger.info("Starting API Gateway...")
    logger.info(f"vLLM URL: {settings.vllm_url}")
    logger.info(f"Home Assistant URL: {settings.ha_url}")
    
    # Initialize HA client
    try:
        ha_client = HomeAssistantClient(settings.ha_url, settings.ha_access_token)
        async with ha_client:
            connected = await ha_client.test_connection()
            if connected:
                logger.info("Successfully connected to Home Assistant")
            else:
                logger.warning("Could not connect to Home Assistant")
    except Exception as e:
        logger.error(f"Error initializing HA client: {e}")
        ha_client = None

    # Start background entity cache refresher
    if ha_client:
        try:
            await start_entity_cache_refresher(ha_client)
        except Exception as e:
            logger.warning(f"Failed to start cache refresher: {e}")

    yield
    
    # Cleanup
    if ha_client:
        try:
            await stop_entity_cache_refresher()
        except Exception as e:
            logger.warning(f"Error stopping cache refresher: {e}")
        if ha_client._session:
            await ha_client._session.close()
    logger.info("Shutting down API Gateway...")


app = FastAPI(
    title="Local LLM API Gateway",
    description="Gateway for local LLM integration with Home Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to HA URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "api-gateway"}


# OpenAI-compatible model discovery endpoint
@app.get("/v1/models")
async def list_models():
    """Minimal OpenAI-compatible model list for Home Assistant / Extended OpenAI discovery"""
    return {
        "object": "list",
        "data": [
            {
                "id": "blaskgpt",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


# OpenAI-compatible chat completion endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint that routes to appropriate handler based on use case
    """
    try:
        # Extract user message from messages array
        # Find the last user message (or system + user combination)
        user_message = None
        system_message = None
        
        for msg in reversed(request.messages):
            if msg.role == "user" and user_message is None:
                # Extract content - handle both string and list formats
                if isinstance(msg.content, str):
                    user_message = msg.content
                elif isinstance(msg.content, list):
                    # Handle content array format (e.g., text blocks)
                    text_parts = [item.get("text", "") for item in msg.content if isinstance(item, dict) and item.get("type") == "text"]
                    user_message = " ".join(text_parts) if text_parts else ""
                break
            elif msg.role == "system" and system_message is None:
                if isinstance(msg.content, str):
                    system_message = msg.content
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in messages array")
        
        # Determine use case if not specified
        if request.use_case:
            use_case = request.use_case
        else:
            use_case = await _detect_use_case(user_message)
        
        # Determine max_tokens: use request value if provided, otherwise use case-specific defaults
        if request.max_tokens is not None:
            max_tokens = request.max_tokens
        else:
            # Use case-specific defaults
            if use_case == "ha_command":
                max_tokens = 200  # Keep low for TTS
            elif use_case == "googling":
                max_tokens = 300
            else:
                max_tokens = 2000  # Higher default for general chat/WebUI
        
        logger.info(f"Processing request: use_case={use_case}, max_tokens={max_tokens}, message={user_message[:50]}...")
        
        # Route to appropriate handler, passing max_tokens
        if use_case == "ha_command":
            response_text = await execute_ha_command(user_message, ha_client, settings, max_tokens=max_tokens)
        elif use_case == "googling":
            response_text = await _handle_googling(user_message, max_tokens=max_tokens)
        elif use_case == "automation_variation":
            response_text = await _handle_automation_variation(user_message, None)
        else:
            # Default: general chat
            response_text = await _handle_general_chat(user_message, max_tokens=max_tokens)
        
        # Build OpenAI-compatible response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())
        
        # Estimate token usage (rough approximation)
        prompt_tokens = len(user_message.split()) * 1.3  # Rough estimate
        completion_tokens = len(response_text.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)
        
        return ChatCompletionResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens
            )
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat completion request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Automation variation endpoint
@app.post("/variations", response_model=VariationResponse)
async def generate_variations(request: VariationRequest):
    """
    Generate variations on automation messages
    """
    try:
        # This will be implemented in automation-variations service
        # For now, return a placeholder
        variations = await _generate_message_variations(
            request.base_message,
            request.count
        )
        return VariationResponse(variations=variations)
    
    except Exception as e:
        logger.error(f"Error generating variations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Webhook endpoint for Home Assistant automations
@app.post("/webhook/variation")
async def webhook_variation(request: Request):
    """
    Webhook endpoint for Home Assistant to request message variations
    """
    try:
        data = await request.json()
        base_message = data.get("message", "")
        count = data.get("count", 1)
        
        variations = await _generate_message_variations(base_message, count)
        
        # Return in format HA expects
        return {"variations": variations}
    
    except Exception as e:
        logger.error(f"Error in webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def _detect_use_case(message: str) -> str:
    """
    Use a fast local keyword route for obvious HA commands, then fall back to LLM classification.
    """
    msg = message.lower().strip()

    ha_fast_patterns = [
        r"\bturn\s+(on|off)\b",
        r"\bswitch\s+(on|off)\b",
        r"\bset\s+.*\b(light|lights|lamp|lamps|volume|brightness|timer)\b",
        r"\b(dim|brighten)\b",
        r"\b(lock|unlock)\b",
        r"\b(play|pause|resume|stop|skip)\b",
        r"\b(volume|mute|unmute)\b",
        r"\b(start|set|cancel|stop)\s+.*\btimer\b",
        r"\badd\s+.*\b(shopping list|shopping|list)\b",
        r"\b(find|ring)\s+my\s+phone\b",
        r"\bwhat\s+(lights|switches|speakers|devices|entities)\b",
    ]

    if any(re.search(pattern, msg) for pattern in ha_fast_patterns):
        logger.info("Fast-classified message as: ha_command")
        return "ha_command"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            classification_prompt = """Classify the user's message into one of these use cases:
- "ha_command": Commands to control Home Assistant devices (lights, switches, locks, speakers, timers, shopping list, etc.) or questions about Home Assistant entities/devices. This includes action commands like "find my phone", "ring my phone", "turn on lights", "play music", "add to shopping list", "start timer", "lock the door", etc. Even if phrased as a question, if it's about controlling or interacting with devices, it's ha_command.
- "googling": Factual questions that require web search and are NOT about controlling devices (e.g., "What is the capital of France?", "How does photosynthesis work?", "What is the speed of light?")
- "automation_variation": Requests to generate variations of automation phrases
- "general": General conversation, greetings, or other non-specific requests

Examples:
- "find my phone" → ha_command (device control)
- "ring my phone" → ha_command (device control)
- "what is the capital of France?" → googling (factual question)
- "turn on the lights" → ha_command (device control)
- "how do I find my phone?" → ha_command (asking how to control a device, not a factual question)

Respond with ONLY the use case name (one word), nothing else."""
            
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "blaskgpt",
                    "messages": [
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": f"/no_think {message}"}
                    ],
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_tokens": 10,  # Just need the use case name
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            use_case = result["choices"][0]["message"]["content"].strip().lower()
            
            # Validate and normalize the response
            valid_use_cases = ["ha_command", "googling", "automation_variation", "general"]
            if use_case in valid_use_cases:
                logger.info(f"LLM classified message as: {use_case}")
                return use_case
            else:
                logger.warning(f"LLM returned unexpected use case: {use_case}, defaulting to general")
                return "general"
    
    except Exception as e:
        logger.error(f"Error in LLM use case detection: {e}, falling back to keyword-based detection")
        # Fallback to simple keyword-based detection
        return _detect_use_case_fallback(message)


def _detect_use_case_fallback(message: str) -> str:
    """
    Fallback keyword-based use case detection if LLM classification fails
    """
    message_lower = message.lower()
    
    # Home Assistant commands - check these first
    ha_keywords = ["shuffle", "play", "add", "set", "timer", "lock", "locks", "unlock", 
                   "turn on", "turn off", "shopping list", "speaker", "speakers",
                   "find my", "find the", "ring my", "ring the", "locate my", "locate the",
                   "list", "show", "entities", "devices"]
    if any(keyword in message_lower for keyword in ha_keywords):
        return "ha_command"
    
    # Googling queries - but exclude device control questions
    question_words = ["what", "who", "where", "when", "why", "how", "can", "is", "are"]
    device_control_indicators = ["my phone", "my device", "the lights", "the lock", "the speaker", 
                                  "my lock", "my lights", "my speaker", "my timer", "my shopping"]
    if any(message_lower.startswith(word) for word in question_words):
        # If it's a question but contains device control indicators, it's likely ha_command
        if any(indicator in message_lower for indicator in device_control_indicators):
            return "ha_command"
        return "googling"
    
    # Default to general chat
    return "general"


async def _handle_ha_command(message: str, context: Optional[Dict] = None) -> str:
    """Compat wrapper delegating to the new HA executor module."""
    if not ha_client:
        return "Home Assistant client is not available. Please check the connection."

    try:
        return await execute_ha_command(message, ha_client, settings)
    except Exception as e:
        logger.error(f"Error in HA command handler: {e}", exc_info=True)
        return f"I encountered an error processing your command: {str(e)}"


def _filter_dummy_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out dummy/helper entities that should not be used for control.
    Entities with 'dummy' in their entity_id or friendly_name are filtered out.
    """
    filtered = []
    for entity in entities:
        entity_id = entity.get("entity_id", "").lower()
        friendly_name = entity.get("attributes", {}).get("friendly_name", "").lower()
        
        # Skip entities with "dummy" in entity_id or friendly_name
        if "dummy" not in entity_id and "dummy" not in friendly_name:
            filtered.append(entity)
        else:
            logger.debug(f"Filtered out dummy entity: {entity.get('entity_id')}")
    
    return filtered


async def _execute_ha_function(function_name: str, args: Dict[str, Any], original_message: Optional[str] = None) -> str:
    """Execute a Home Assistant function"""
    try:
        async with ha_client:
            if function_name == "search_music":
                query = args.get("query", "")
                media_type = args.get("media_type", "any")
                limit = args.get("limit", 10)
                
                if not query:
                    return "Error: No search query provided"
                
                try:
                    # Find a Music Assistant media player entity to get config_entry_id
                    speakers = await ha_client.find_speakers()
                    speakers = _filter_dummy_entities(speakers)
                    
                    if not speakers:
                        return "Error: No media player entities found. Please ensure Music Assistant is configured."
                    
                    # Get config_entry_id from the first Music Assistant entity
                    # We need to get it from the device registry via the entity's device_id
                    ma_entity = speakers[0]["entity_id"]
                    config_entry_id = None
                    
                    try:
                        # Get entity state to find device_id
                        entity_state = await ha_client.get_entity(ma_entity)
                        device_id = entity_state.get("attributes", {}).get("device_id")
                        
                        if device_id:
                            # Get device info from device registry
                            try:
                                device_info = await ha_client._request("GET", f"config/device_registry/{device_id}")
                                # Get config_entries from device
                                config_entries = device_info.get("config_entries", [])
                                if config_entries:
                                    # Get the first config entry (usually Music Assistant)
                                    config_entry_id = config_entries[0]
                                    logger.info(f"Found config_entry_id: {config_entry_id} for entity {ma_entity}")
                            except Exception as e:
                                logger.warning(f"Could not get config_entry_id from device registry: {e}")
                    except Exception as e:
                        logger.warning(f"Could not get device_id from entity {ma_entity}: {e}")
                    
                    if not config_entry_id:
                        return "Error: Could not find Music Assistant config_entry_id. Please ensure Music Assistant is properly configured and the entity is associated with a Music Assistant device."
                    
                    # Call music_assistant.search service
                    # Required: config_entry_id (or try without it), name (the search query)
                    search_data = {
                        "name": query  # The search query goes in 'name' parameter
                    }
                    
                    # Add config_entry_id if we found it
                    if config_entry_id:
                        search_data["config_entry_id"] = config_entry_id
                    
                    # Add optional parameters if provided
                    if media_type and media_type != "any":
                        search_data["media_type"] = media_type
                    if limit:
                        search_data["limit"] = limit
                    
                    result = await ha_client.call_service(
                        "music_assistant",
                        "search",
                        return_response=True,
                        **search_data
                    )
                    
                    # Parse search response
                    # Response structure may vary, but typically contains items with uri, name, etc.
                    items = []
                    if isinstance(result, dict):
                        service_response = result.get("service_response", {})
                        if isinstance(service_response, dict):
                            # Check for items in various possible locations
                            items = service_response.get("items", service_response.get("results", []))
                        elif isinstance(service_response, list):
                            items = service_response
                    elif isinstance(result, list):
                        items = result
                    
                    if not items:
                        return f"No music found for query: {query}. Try a different search term or check that Music Assistant has music available."
                    
                    # Format results for the LLM
                    formatted_results = []
                    for item in items[:limit]:
                        if isinstance(item, dict):
                            # Music Assistant search results typically have uri, name, artist, etc.
                            uri = item.get("uri", item.get("media_content_id", ""))
                            name = item.get("name", item.get("title", "Unknown"))
                            artist = item.get("artist", item.get("artist_name", ""))
                            item_type = item.get("media_type", item.get("type", "music"))
                            
                            if artist:
                                display_name = f"{name} by {artist}"
                            else:
                                display_name = name
                            
                            formatted_results.append(f"{display_name} ({item_type}) - URI: {uri}")
                        else:
                            formatted_results.append(str(item))
                    
                    result_text = f"Found {len(items)} result(s) for '{query}':\n" + "\n".join(formatted_results)
                    return result_text
                    
                except Exception as e:
                    logger.error(f"Error searching music: {e}", exc_info=True)
                    error_str = str(e)
                    
                    if "400" in error_str or "Bad Request" in error_str:
                        return f"Error: Music Assistant search returned 400 Bad Request. Check that config_entry_id is correct and 'name' parameter is provided. Error: {str(e)}"
                    elif "404" in error_str or "Not Found" in error_str:
                        return f"Error: Music Assistant search service not found (404). Please check that Music Assistant is installed and configured in Home Assistant."
                    else:
                        return f"Error searching for music: {str(e)}"
            
            elif function_name == "play_media_on_speakers":
                entity_names = args.get("entity_names", [])
                media_content_id = args.get("media_content_id", "")
                media_content_type = args.get("media_content_type", "music")
                
                # Handle "all" speakers
                if "all" in [name.lower() for name in entity_names]:
                    speakers = await ha_client.find_speakers()
                    # Filter out dummy entities
                    speakers = _filter_dummy_entities(speakers)
                    entity_ids = [s["entity_id"] for s in speakers]
                else:
                    # Validate that all entity IDs exist before proceeding
                    entity_ids = []
                    invalid_entities = []
                    
                    for entity_name in entity_names:
                        # Check if it looks like an entity_id (has a dot) or is just a name
                        if "." in entity_name:
                            # It's an entity_id, verify it exists
                            try:
                                entity_state = await ha_client.get_entity(entity_name)
                                entity_ids.append(entity_name)
                            except Exception as e:
                                logger.warning(f"Entity {entity_name} not found: {e}")
                                invalid_entities.append(entity_name)
                        else:
                            # It's not a proper entity_id format - this shouldn't happen if LLM followed instructions
                            invalid_entities.append(entity_name)
                    
                    # If any entities are invalid, return an error that prompts the LLM to search
                    if invalid_entities:
                        return f"Error: The following speaker entities were not found: {', '.join(invalid_entities)}. You must first call list_available_entities with domain='media_player' to get the exact entity_id from Home Assistant. Do not guess or construct entity_ids."
                
                if not entity_ids:
                    return "Error: No valid speaker entities found. You must first call list_available_entities with domain='media_player' to get available speakers."
                
                result = await ha_client.call_service(
                    "media_player",
                    "play_media",
                    entity_id=entity_ids,
                    media_content_id=media_content_id,
                    media_content_type=media_content_type
                )
                return f"Playing {media_content_id} on speakers: {', '.join(entity_ids)}"
            
            elif function_name == "add_item_to_shopping_list":
                item_name = args.get("item_name", "")
                if not item_name:
                    return "Error: No item name provided"
                
                # Use todo.add_item service with the Google Keep shopping list entity
                entity_id = "todo.google_keep_shopping"
                logger.info(f"Calling todo.add_item service with entity_id='{entity_id}', item='{item_name}'")
                try:
                    result = await ha_client.call_service(
                        "todo",
                        "add_item",
                        entity_id=entity_id,
                        item=item_name
                    )
                    logger.info(f"todo.add_item service call result: {result}")
                    
                    # Check if result indicates success
                    if isinstance(result, list):
                        # Service call succeeded
                        logger.info(f"Service call completed. Item '{item_name}' should be added to shopping list.")
                        return f"Added {item_name} to shopping list"
                    elif isinstance(result, dict):
                        # Check for error status
                        if result.get("status") == "error" or "error" in result:
                            logger.error(f"Error response from todo.add_item: {result}")
                            return f"Error: Failed to add {item_name} to shopping list. Response: {result}"
                        else:
                            return f"Added {item_name} to shopping list"
                    else:
                        logger.warning(f"Unexpected result type from todo.add_item: {type(result)}, value: {result}")
                        return f"Added {item_name} to shopping list"
                        
                except Exception as e:
                    # Check if it's an HTTP error
                    error_str = str(e)
                    if "404" in error_str or "Not Found" in error_str:
                        logger.error(f"Service not found: {e}")
                        return f"Error: todo.add_item service not found or entity '{entity_id}' not available. Please check Home Assistant configuration."
                    elif "401" in error_str or "Unauthorized" in error_str:
                        logger.error(f"Authentication error: {e}")
                        return f"Error: Authentication failed when calling Home Assistant service."
                    else:
                        logger.error(f"Exception calling todo.add_item: {e}", exc_info=True)
                        return f"Error: Failed to add {item_name} to shopping list: {str(e)}"
            
            elif function_name == "get_shopping_list_items":
                # Use todo.get_items service to retrieve items from the shopping list
                entity_id = "todo.google_keep_shopping"
                logger.info(f"Getting shopping list items from entity '{entity_id}' using todo.get_items service")
                try:
                    # Call todo.get_items service with return_response=true
                    # Use entity_id directly (not in target), and filter for needs_action items only
                    service_data = {
                        "entity_id": entity_id,
                        "status": ["needs_action"]  # Only get incomplete items for shopping list
                    }
                    logger.info(f"Calling todo.get_items with data: {json.dumps(service_data)}")
                    
                    result = await ha_client.call_service(
                        "todo",
                        "get_items",
                        return_response=True,  # Required for services that return data
                        **service_data
                    )
                    logger.info(f"todo.get_items service call result type: {type(result)}")
                    logger.debug(f"todo.get_items service call result: {json.dumps(result, indent=2)}")
                    
                    # Parse response: response_json["service_response"]["todo.google_keep_shopping"]["items"]
                    items = []
                    if isinstance(result, dict):
                        service_response = result.get("service_response", {})
                        if entity_id in service_response:
                            entity_data = service_response[entity_id]
                            if isinstance(entity_data, dict) and "items" in entity_data:
                                items = entity_data["items"]
                                logger.info(f"Found {len(items)} items in service response")
                    
                    if not items:
                        return "The shopping list is empty."
                    
                    # Format items for response
                    # Items are dicts with 'summary', 'uid', and 'status' fields
                    item_list = []
                    for item in items:
                        if isinstance(item, dict):
                            # Extract the summary (item name)
                            item_name = item.get("summary") or item.get("name") or item.get("item") or str(item)
                        else:
                            item_name = str(item)
                        item_list.append(item_name)
                    
                    item_count = len(item_list)
                    items_text = "\n".join(f"- {item}" for item in item_list)
                    
                    # Return format that emphasizes the full list
                    return f"Shopping list ({item_count} item{'s' if item_count != 1 else ''}):\n{items_text}"
                    
                except Exception as e:
                    logger.error(f"Exception getting shopping list items: {e}", exc_info=True)
                    error_str = str(e)
                    if "404" in error_str or "Not Found" in error_str:
                        return f"Error: Shopping list entity '{entity_id}' not found. Please check Home Assistant configuration."
                    else:
                        return f"Error: Failed to get shopping list items: {str(e)}"
            
            elif function_name == "start_timer":
                duration_minutes = args.get("duration_minutes", 0)
                name = args.get("name")
                
                if duration_minutes == 0:
                    return "Error: duration_minutes is required for start_timer"
                
                # Convert minutes to HH:MM:SS format for the script
                duration_hhmmss = _minutes_to_hhmmss(duration_minutes)
                
                # Call the set_next_available_timer script instead of timer.start directly
                result = await ha_client.call_service(
                    "script",
                    "turn_on",
                    entity_id="script.set_next_available_timer",
                    duration=duration_hhmmss
                )
                
                # Note: The script doesn't support name parameter, so we ignore it
                # The script will pick the first available timer (timer1-3) automatically
                timer_name = name or "timer"
                return f"Started {timer_name} for {duration_minutes} minutes"
            
            elif function_name == "get_entity_state":
                entity_id = args.get("entity_id", "")
                entity = await ha_client.get_entity(entity_id)
                return f"Entity {entity_id} state: {entity.get('state', 'unknown')}"
            
            elif function_name == "find_phone_blasxel_6":
                result = await ha_client.call_service(
                    "script",
                    "find_phone_blasxel_6"
                )
                return "Started find phone script for Blasxel 6"
            
            elif function_name == "list_available_entities":
                domain = args.get("domain")
                query = args.get("query", "")
                if query:
                    entities = await ha_client.search_entities(query, domain)
                else:
                    entities = await ha_client.get_entities()
                    if domain:
                        entities = [e for e in entities if e["entity_id"].startswith(f"{domain}.")]
                
                # Filter out dummy entities
                entities = _filter_dummy_entities(entities)
                
                # Format response with entity IDs and friendly names
                entity_list = []
                for e in entities:
                    entity_id = e.get("entity_id", "")
                    friendly_name = e.get("attributes", {}).get("friendly_name", entity_id)
                    entity_list.append(f"{entity_id} ({friendly_name})")
                
                return f"Found {len(entities)} entities:\n" + "\n".join(entity_list[:50])  # Limit to 50 for readability
            
            elif function_name in ["turn_on_entity", "turn_off_entity"]:
                entity_id = args.get("entity_id", "")
                original_entity_id = entity_id
                
                logger.info(f"Executing {function_name} with entity_id: '{entity_id}'")
                
                # Check if this looks like a guessed/constructed entity_id
                # If the original message contains natural language that matches the entity_id pattern,
                # it's likely a guess and we should reject it
                if original_message:
                    message_lower = original_message.lower()
                    # Extract the descriptive part from entity_id (e.g., "den_lights" from "light.den_lights")
                    if "." in entity_id:
                        entity_desc = entity_id.split(".", 1)[1].replace("_", " ").lower()
                        # If the entity description appears in the original message, it might be a guess
                        # But we need to be careful - if it's an exact match from list_available_entities, that's fine
                        # The real issue is when the LLM constructs entity_ids without calling list_available_entities first
                        # We'll check if the entity exists first, and if not, we'll do semantic matching
                        # But we should log a warning if it looks like a guess
                        if entity_desc in message_lower and not any(word in message_lower for word in ["list", "show", "what", "available", "entities"]):
                            logger.warning(f"Entity_id '{entity_id}' looks like it might be guessed from message '{original_message}'. Will verify it exists first.")
                
                # First, try to verify the entity exists
                entity_exists = False
                try:
                    entity_state = await ha_client.get_entity(entity_id)
                    entity_exists = True
                    domain = entity_id.split(".")[0] if "." in entity_id else "switch"
                    logger.info(f"Entity {entity_id} exists, current state: {entity_state.get('state', 'unknown')}")
                except Exception as e:
                    logger.info(f"Entity {entity_id} not found, attempting to resolve from natural language")
                    entity_exists = False
                
                # If entity doesn't exist or doesn't look like a proper entity_id, try to resolve it
                if not entity_exists:
                    resolved = False
                    
                    # Determine the domain from the entity_id if possible
                    if "." in entity_id:
                        likely_domain = entity_id.split(".")[0]
                        search_term = entity_id.split(".", 1)[1].replace("_", " ").lower()
                    else:
                        # Try to infer domain from original message
                        user_desc = original_message.lower() if original_message else ""
                        if "lock" in user_desc or "unlock" in user_desc:
                            likely_domain = "lock"
                        elif "light" in user_desc:
                            likely_domain = "light"
                        elif "switch" in user_desc:
                            likely_domain = "switch"
                        else:
                            likely_domain = None
                        search_term = entity_id.lower().replace("_", " ")
                    
                    # If we have a likely domain, use semantic matching with all entities in that domain
                    # This is the preferred approach - get all entities in the domain and match semantically
                    # For lights, also check the switch domain as many light switches are in the switch domain
                    if likely_domain:
                        # If user mentioned "light" or "lights", check both light and switch domains
                        check_domains = [likely_domain]
                        if likely_domain == "light" or (original_message and "light" in original_message.lower()):
                            if "switch" not in check_domains:
                                check_domains.append("switch")
                            logger.info(f"User mentioned lights - will check both 'light' and 'switch' domains")
                        
                        logger.info(f"Attempting semantic matching for '{search_term}' in domains {check_domains}")
                        try:
                            # Get all entities in the relevant domains
                            all_entities = await ha_client.get_entities()
                            domain_entities = []
                            for domain_to_check in check_domains:
                                domain_entities.extend([e for e in all_entities if e["entity_id"].startswith(f"{domain_to_check}.")])
                            
                            # Filter out dummy entities
                            domain_entities = _filter_dummy_entities(domain_entities)
                            
                            if domain_entities:
                                # Use LLM for semantic matching
                                entity_list = []
                                for e in domain_entities:
                                    entity_id_val = e.get("entity_id", "")
                                    friendly_name = e.get("attributes", {}).get("friendly_name", entity_id_val)
                                    entity_list.append(f"{entity_id_val} ({friendly_name})")
                                
                                entity_list_str = "\n".join(entity_list)
                                domains_str = " and ".join(check_domains)
                                
                                async with httpx.AsyncClient(timeout=10.0) as llm_client:
                                    match_response = await llm_client.post(
                                        f"{settings.vllm_url}/v1/chat/completions",
                                        headers={
                                            "Authorization": f"Bearer {settings.vllm_api_key}",
                                            "Content-Type": "application/json"
                                        },
                                        json={
                                            "model": "blaskgpt",
                                            "messages": [
                                                {"role": "system", "content": "You match user descriptions to Home Assistant entities using semantic understanding. Match based on meaning: vehicle names/brands match 'car', room names match locations, device types match functions. Return ONLY the exact entity_id that best matches, nothing else."},
                                                {"role": "user", "content": f"User said: '{original_message if original_message else search_term}'\n\nAvailable {domains_str} entities:\n{entity_list_str}\n\nWhich entity_id semantically matches what the user wants to control? Use semantic understanding - for example, 'car' matches vehicle names like 'rav4', 'toyota', etc., 'den lights' matches entities with 'den' in the name. Return only the entity_id."}
                                            ],
                                            "temperature": 0.1,
                                            "max_tokens": 100
                                        }
                                    )
                                    match_response.raise_for_status()
                                    match_result = match_response.json()
                                    matched_entity_id = match_result["choices"][0]["message"]["content"].strip()
                                    
                                    # Clean up the response
                                    matched_entity_id = matched_entity_id.strip('"\'`')
                                    entity_id_match = re.search(r'([a-z_]+\.\S+)', matched_entity_id)
                                    if entity_id_match:
                                        matched_entity_id = entity_id_match.group(1)
                                    
                                    # Verify the matched entity exists
                                    candidate_ids = [e.get("entity_id", "") for e in domain_entities]
                                    if matched_entity_id in candidate_ids:
                                        entity_id = matched_entity_id
                                        # Extract domain from matched entity_id
                                        domain = matched_entity_id.split(".")[0] if "." in matched_entity_id else likely_domain
                                        logger.info(f"Semantically matched '{search_term}' to entity_id '{entity_id}' in domain '{domain}'")
                                        resolved = True
                                    else:
                                        logger.warning(f"LLM returned entity_id not in candidates: {matched_entity_id}. Candidates: {candidate_ids[:3]}")
                                        # If semantic matching returned invalid entity, don't fall back to literal search
                                        # Return error instead
                                        error_msg = f"Error: Semantic matching failed to find valid {domains_str} entity matching '{original_message if original_message else search_term}'"
                                        logger.warning(error_msg)
                                        return error_msg
                        except Exception as e:
                            logger.error(f"Error in semantic entity matching: {e}", exc_info=True)
                            # If semantic matching fails with a known domain, return error instead of falling back
                            error_msg = f"Error: Could not semantically match '{original_message if original_message else search_term}' to a {check_domains} entity: {str(e)}"
                            return error_msg
                    
                    # Only fallback to literal search if semantic matching didn't work AND we have a domain
                    # If we have a domain, we should prefer semantic matching over literal matching
                    if not resolved:
                        # Extract search terms from the entity_id
                        if "." in original_entity_id:
                            search_query = original_entity_id.split(".", 1)[1].replace("_", " ").lower()
                            search_queries = [search_query, original_entity_id.lower().replace("_", " ")]
                        else:
                            search_query = original_entity_id.lower()
                            search_queries = [search_query]
                        
                        # Try the likely domain first with literal search
                        if likely_domain:
                            for query in search_queries:
                                entities = await ha_client.search_entities(query, domain=likely_domain)
                                entities = _filter_dummy_entities(entities)
                                if entities:
                                    entity_id = entities[0]["entity_id"]
                                    domain = likely_domain
                                    logger.info(f"Resolved '{original_entity_id}' to entity_id '{entity_id}' via literal search in domain '{likely_domain}'")
                                    resolved = True
                                    break
                        
                        # Try other common domains
                        if not resolved:
                            for test_domain in ["light", "switch", "fan", "climate", "cover", "lock"]:
                                if test_domain == likely_domain:
                                    continue  # Already tried
                                for query in search_queries:
                                    entities = await ha_client.search_entities(query, domain=test_domain)
                                    entities = _filter_dummy_entities(entities)
                                    if entities:
                                        entity_id = entities[0]["entity_id"]
                                        domain = test_domain
                                        logger.info(f"Resolved '{original_entity_id}' to entity_id '{entity_id}' via search in domain '{test_domain}'")
                                        resolved = True
                                        break
                                if resolved:
                                    break
                        
                        # Last resort: broad search (but ONLY if we don't have a domain - if we have a domain, semantic matching should have worked)
                        if not resolved and not likely_domain:
                            for query in search_queries:
                                entities = await ha_client.search_entities(query)
                                entities = _filter_dummy_entities(entities)
                                if entities:
                                    entity_id = entities[0]["entity_id"]
                                    domain = entity_id.split(".")[0] if "." in entity_id else "switch"
                                    logger.info(f"Resolved '{original_entity_id}' to entity_id '{entity_id}' via broad search")
                                    resolved = True
                                    break
                    
                    # If we still haven't resolved and we had a domain, the semantic matching should have worked
                    # If semantic matching failed, don't fall back to literal search that might match wrong entities
                    # Instead, return an error
                    if not resolved and likely_domain:
                        error_msg = f"Error: Could not find {likely_domain} entity matching '{original_message if original_message else original_entity_id}'. Semantic matching failed."
                        logger.warning(error_msg)
                        return error_msg
                    
                    # If we don't have a domain, try semantic matching across all domains as last resort
                    if not resolved:
                        # Last resort: Use LLM for semantic entity matching
                        # Use the original user message description, not the constructed entity_id
                        user_description = original_message if original_message else original_entity_id
                        logger.info(f"Attempting semantic entity matching for user description: '{user_description}'")
                        try:
                            # Determine likely domain from the entity_id or function name
                            if "." in original_entity_id:
                                likely_domain = original_entity_id.split(".")[0]
                            else:
                                # Try to infer from function name or user message
                                likely_domain = "lock" if "lock" in user_description.lower() else None
                            
                            # Get all entities in the likely domain(s)
                            all_candidates = []
                            domains_to_try = [likely_domain] if likely_domain else ["lock", "light", "switch", "fan", "climate", "cover"]
                            
                            for test_domain in domains_to_try:
                                if test_domain:
                                    entities = await ha_client.get_entities()
                                    domain_entities = [e for e in entities if e["entity_id"].startswith(f"{test_domain}.")]
                                    # Filter out dummy entities
                                    domain_entities = _filter_dummy_entities(domain_entities)
                                    all_candidates.extend(domain_entities)
                            
                            if all_candidates:
                                # Format entities for LLM
                                entity_list = []
                                for e in all_candidates:
                                    entity_id_val = e.get("entity_id", "")
                                    friendly_name = e.get("attributes", {}).get("friendly_name", entity_id_val)
                                    entity_list.append(f"{entity_id_val} ({friendly_name})")
                                
                                entity_list_str = "\n".join(entity_list)
                                
                                # Use LLM to semantically match using the original user description
                                async with httpx.AsyncClient(timeout=10.0) as llm_client:
                                    match_response = await llm_client.post(
                                        f"{settings.vllm_url}/v1/chat/completions",
                                        headers={
                                            "Authorization": f"Bearer {settings.vllm_api_key}",
                                            "Content-Type": "application/json"
                                        },
                                        json={
                                            "model": "blaskgpt",
                                            "messages": [
                                                {"role": "system", "content": "You match user descriptions to Home Assistant entities using semantic understanding. Match based on meaning: vehicle names/brands match 'car', room names match locations, device types match functions. Return ONLY the exact entity_id that best matches, nothing else."},
                                                {"role": "user", "content": f"User said: '{user_description}'\n\nAvailable entities:\n{entity_list_str}\n\nWhich entity_id semantically matches what the user wants to control? Return only the entity_id."}
                                            ],
                                            "temperature": 0.1,
                                            "max_tokens": 100
                                        }
                                    )
                                    match_response.raise_for_status()
                                    match_result = match_response.json()
                                    matched_entity_id = match_result["choices"][0]["message"]["content"].strip()
                                    
                                    # Clean up the response (remove quotes, extra text)
                                    matched_entity_id = matched_entity_id.strip('"\'`')
                                    # Extract entity_id if LLM added extra text
                                    entity_id_match = re.search(r'([a-z_]+\.\S+)', matched_entity_id)
                                    if entity_id_match:
                                        matched_entity_id = entity_id_match.group(1)
                                    
                                    # Verify the matched entity exists
                                    candidate_ids = [e.get("entity_id", "") for e in all_candidates]
                                    if matched_entity_id in candidate_ids:
                                        entity_id = matched_entity_id
                                        domain = entity_id.split(".")[0] if "." in entity_id else "switch"
                                        logger.info(f"Semantically matched '{user_description}' to entity_id '{entity_id}'")
                                        resolved = True
                                    else:
                                        logger.warning(f"LLM returned entity_id not in candidates: {matched_entity_id}. Candidates: {candidate_ids[:5]}")
                        except Exception as e:
                            logger.error(f"Error in semantic entity matching: {e}", exc_info=True)
                    
                    if not resolved:
                        error_msg = f"Error: Could not find entity matching '{original_entity_id}'. Please check the entity name or use list_available_entities to find the correct entity_id."
                        logger.warning(error_msg)
                        return error_msg
                    
                    # Verify the resolved entity exists
                    try:
                        entity_state = await ha_client.get_entity(entity_id)
                        logger.info(f"Resolved entity {entity_id} current state: {entity_state.get('state', 'unknown')}")
                    except Exception as e:
                        error_msg = f"Error: Resolved entity {entity_id} not found in Home Assistant: {str(e)}"
                        logger.error(error_msg)
                        return error_msg
                else:
                    domain = entity_id.split(".")[0] if "." in entity_id else "switch"
                
                # For lock domain, use lock/unlock services; for others, use turn_on/turn_off
                if domain == "lock":
                    service = "lock" if function_name == "turn_on_entity" else "unlock"
                else:
                    service = "turn_on" if function_name == "turn_on_entity" else "turn_off"
                logger.info(f"Calling Home Assistant service: {domain}.{service} with entity_id={entity_id}")
                
                try:
                    result = await ha_client.call_service(domain, service, entity_id=entity_id)
                    logger.info(f"Home Assistant response: {result}")
                    
                    # Check if result indicates success
                    # Home Assistant typically returns a list of state changes
                    if isinstance(result, list) and len(result) > 0:
                        # Verify the state changed
                        new_state = await ha_client.get_entity(entity_id)
                        new_state_value = new_state.get('state', 'unknown')
                        logger.info(f"Entity {entity_id} new state: {new_state_value}")
                        
                        # Return success message with state confirmation
                        return f"{service.replace('_', ' ').title()} {entity_id} (state: {new_state_value})"
                    else:
                        # Response might be empty or different format
                        logger.warning(f"Unexpected response format from Home Assistant: {result}")
                        return f"{service.replace('_', ' ').title()} {entity_id} (response: {result})"
                except Exception as e:
                    error_msg = f"Error calling Home Assistant service: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return error_msg
            
            else:
                return f"Unknown function: {function_name}"
    
    except Exception as e:
        logger.error(f"Error executing HA function {function_name}: {e}", exc_info=True)
        return f"Error executing function: {str(e)}"


async def _handle_googling(message: str, max_tokens: int = 300) -> str:
    """
    Handle googling/search queries with web search
    """
    try:
        # Perform web search
        search_service = get_search_service()
        search_results = await search_service.search(message, max_results=5)
        
        if not search_results:
            # Check if it's a rate limit issue by checking logs or providing a helpful message
            logger.warning(f"No search results for query: {message}")
            return "I'm currently unable to perform web searches due to rate limiting. Please try again in a few moments, or rephrase your question."
        
        # Format search context
        search_context = search_service.format_search_context(search_results)
        
        # Load googling system prompt
        # Try /app/prompts/ first (Docker container path), then /opt/llm/prompts/ (host path)
        prompt_paths = [
            "/app/prompts/googling_system.txt",
            "/opt/llm/prompts/googling_system.txt"
        ]
        system_prompt = None
        for prompt_path in prompt_paths:
            try:
                with open(prompt_path, "r") as f:
                    system_prompt = f.read()
                break
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers factual questions using web search results. Provide concise, direct answers."
        
        # Call LLM with search context
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "blaskgpt",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {message}\n\nSearch Results:\n{search_context}\n\nAnswer the question concisely:"}
                    ],
                    "temperature": 0.3,  # Lower temperature for factual answers
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            result = response.json()
            message_obj = result["choices"][0]["message"]
            content = message_obj.get("content") or ""
            if not content:
                logger.warning(f"LLM returned empty content. message={message_obj}")
            return content
    
    except Exception as e:
        logger.error(f"Error in googling handler: {e}", exc_info=True)
        return f"I encountered an error while searching: {str(e)}"


async def _handle_automation_variation(message: str, context: Optional[Dict] = None) -> str:
    """
    Handle automation variation generation
    """
    try:
        count = context.get("count", 1) if context else 1
        
        VarSettings = automation_module.Settings
        var_settings = VarSettings()
        variations = await gen_variations(message, count, var_settings)
        
        if len(variations) == 1:
            return variations[0]
        else:
            return "\n".join([f"{i+1}. {v}" for i, v in enumerate(variations)])
    
    except Exception as e:
        logger.error(f"Error in automation variation handler: {e}", exc_info=True)
        return message  # Fallback to original message


async def _handle_general_chat(message: str, max_tokens: int = 2000) -> str:
    """
    Handle general chat requests using vLLM with conversational system prompt
    """
    try:
        # Load conversational system prompt
        prompt_paths = [
            "/app/prompts/conversational_system.txt",
            "/opt/llm/prompts/conversational_system.txt"
        ]
        system_prompt = None
        for prompt_path in prompt_paths:
            try:
                with open(prompt_path, "r") as f:
                    system_prompt = f.read()
                break
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        if system_prompt is None:
            system_prompt = "You are a helpful, conversational assistant. Engage naturally with the user in a friendly and informative manner."
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "blaskgpt",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"/no_think {message}"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            message_obj = result["choices"][0]["message"]
            content = message_obj.get("content") or ""
            if not content:
                logger.warning(f"LLM returned empty content. message={message_obj}")
            return content
    
    except Exception as e:
        logger.error(f"Error calling vLLM: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")


async def _generate_message_variations(base_message: str, count: int) -> list[str]:
    """
    Generate variations of a message using LLM
    """
    try:
        VarSettings = automation_module.Settings
        var_settings = VarSettings()
        variations = await gen_variations(base_message, count, var_settings)
        return variations
    except Exception as e:
        logger.error(f"Error generating variations: {e}", exc_info=True)
        return [base_message] * count


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
