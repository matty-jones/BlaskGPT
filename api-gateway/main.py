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
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.ha_functions import get_function_schemas

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


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat/completion requests"""
    message: str
    use_case: Optional[str] = None  # "ha_command", "googling", "automation_variation"
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat/completion responses"""
    response: str
    use_case: str
    metadata: Optional[Dict[str, Any]] = None


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
    
    yield
    
    # Cleanup
    if ha_client and ha_client._session:
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


# Chat/completion endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that routes to appropriate handler based on use case
    """
    try:
        # Determine use case if not specified
        if request.use_case:
            use_case = request.use_case
        else:
            use_case = await _detect_use_case(request.message)
        
        logger.info(f"Processing request: use_case={use_case}, message={request.message[:50]}...")
        
        # Route to appropriate handler
        if use_case == "ha_command":
            response_text = await _handle_ha_command(request.message, request.context)
        elif use_case == "googling":
            response_text = await _handle_googling(request.message)
        elif use_case == "automation_variation":
            response_text = await _handle_automation_variation(request.message, request.context)
        else:
            # Default: general chat
            response_text = await _handle_general_chat(request.message)
        
        return ChatResponse(
            response=response_text,
            use_case=use_case,
            metadata={"model": "Qwen/Qwen2.5-7B-Instruct-AWQ"}
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
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
    Use LLM to intelligently detect the use case based on message content.
    This is more flexible than keyword matching and handles edge cases better.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            classification_prompt = """Classify the user's message into one of these use cases:
- "ha_command": Commands to control Home Assistant devices (lights, switches, locks, speakers, timers, shopping list, etc.) or questions about Home Assistant entities/devices
- "googling": Factual questions that require web search (e.g., "What is the capital of France?", "How does photosynthesis work?")
- "automation_variation": Requests to generate variations of automation phrases
- "general": General conversation, greetings, or other non-specific requests

Respond with ONLY the use case name (one word), nothing else."""
            
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "messages": [
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": message}
                    ],
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "max_tokens": 10  # Just need the use case name
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
    
    # Home Assistant commands
    ha_keywords = ["shuffle", "play", "add", "set", "timer", "lock", "locks", "unlock", 
                   "turn on", "turn off", "shopping list", "speaker", "speakers",
                   "what", "list", "show", "entities", "devices"]
    if any(keyword in message_lower for keyword in ha_keywords):
        return "ha_command"
    
    # Googling queries
    question_words = ["what", "who", "where", "when", "why", "how", "can", "is", "are"]
    if any(message_lower.startswith(word) for word in question_words):
        return "googling"
    
    # Default to general chat
    return "general"


async def _handle_ha_command(message: str, context: Optional[Dict] = None) -> str:
    """
    Handle Home Assistant command requests with function calling
    """
    if not ha_client:
        return "Home Assistant client is not available. Please check the connection."
    
    try:
        # Load HA command system prompt
        try:
            with open("/opt/llm/prompts/ha_command_system.txt", "r") as f:
                system_prompt = f.read()
        except Exception:
            system_prompt = "You are a helpful assistant that controls Home Assistant devices through natural language commands."
        
        # Get function schemas
        functions = get_function_schemas()
        
        # First call: Get function call from LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Convert functions to tools format (vLLM requires tools, not functions)
            tools = functions  # The format is compatible
            
            request_payload = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "tools": tools,  # Use tools instead of functions
                "tool_choice": "auto",  # Use tool_choice instead of function_call
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json=request_payload
            )
            response.raise_for_status()
            result = response.json()
            message_obj = result["choices"][0]["message"]
            
            # Check if LLM wants to call a function (handle both formats)
            tool_calls = message_obj.get("tool_calls")
            function_call = message_obj.get("function_call")
            content = message_obj.get("content", "")
            
            # Handle XML tool call format from qwen3_xml parser
            import re
            xml_tool_call_match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', content, re.DOTALL)
            if xml_tool_call_match:
                try:
                    tool_call_data = json.loads(xml_tool_call_match.group(1))
                    function_name = tool_call_data.get("name")
                    function_args = tool_call_data.get("arguments", {})
                    logger.info(f"Found XML tool call: {function_name} with args {function_args}")
                    
                    # Execute the function (pass original message for semantic matching)
                    execution_result = await _execute_ha_function(function_name, function_args, original_message=message)
                    
                    logger.info(f"Function execution result: {execution_result}")
                    
                    # Check if execution result is an error - if so, return it directly
                    if isinstance(execution_result, str) and (execution_result.startswith("Error:") or execution_result.startswith("Unknown function:")):
                        logger.warning(f"Function execution returned error: {execution_result}")
                        return execution_result
                    
                    # If list_available_entities was called with a query and found 0 results, 
                    # automatically retry without query to get all entities for semantic matching
                    if function_name == "list_available_entities" and "query" in function_args and "Found 0 entities" in str(execution_result):
                        logger.info("list_available_entities with query returned 0 results, retrying without query for semantic matching")
                        domain = function_args.get("domain")
                        retry_args = {"domain": domain} if domain else {}
                        execution_result = await _execute_ha_function("list_available_entities", retry_args, original_message=message)
                        logger.info(f"Retry result: {execution_result}")
                        
                        # Now let the LLM do semantic matching with the full list
                        response2 = await client.post(
                            f"{settings.vllm_url}/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {settings.vllm_api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": message},
                                    {"role": "assistant", "content": content},
                                    {"role": "tool", "name": "list_available_entities", "content": str(execution_result)},
                                    {"role": "user", "content": f"From the list above, use semantic understanding to match the user's request to the correct entity. Then call the appropriate control function (turn_on_entity or turn_off_entity) with the exact entity_id."}
                                ],
                                "tools": tools,
                                "tool_choice": "required",
                                "temperature": 0.3,
                                "max_tokens": 200
                            }
                        )
                        response2.raise_for_status()
                        result2 = response2.json()
                        message_obj2 = result2["choices"][0]["message"]
                        
                        # Check if LLM made another function call
                        tool_calls2 = message_obj2.get("tool_calls")
                        if tool_calls2 and len(tool_calls2) > 0:
                            tool_call2 = tool_calls2[0]
                            function_name2 = tool_call2["function"]["name"]
                            function_args_str2 = tool_call2["function"]["arguments"]
                            if isinstance(function_args_str2, str):
                                function_args2 = json.loads(function_args_str2)
                            else:
                                function_args2 = function_args_str2
                            
                            logger.info(f"LLM selected entity: {function_name2} with args: {function_args2}")
                            execution_result2 = await _execute_ha_function(function_name2, function_args2, original_message=message)
                            
                            # Check if execution result is an error - if so, return it directly
                            if isinstance(execution_result2, str) and (execution_result2.startswith("Error:") or execution_result2.startswith("Unknown function:")):
                                logger.warning(f"Function execution returned error: {execution_result2}")
                                return execution_result2
                            
                            # Final confirmation
                            response3 = await client.post(
                                f"{settings.vllm_url}/v1/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {settings.vllm_api_key}",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": message},
                                        {"role": "assistant", "content": None, "tool_calls": tool_calls2},
                                        {"role": "tool", "name": function_name2, "content": str(execution_result2), "tool_call_id": tool_call2["id"]},
                                        {"role": "user", "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries, no offers of help, no additional information."}
                                    ],
                                    "tools": tools,
                                    "temperature": 0.3,
                                    "max_tokens": 50
                                }
                            )
                            response3.raise_for_status()
                            result3 = response3.json()
                            return result3["choices"][0]["message"]["content"]
                    
                    # Second call: Get natural language response about the action
                    response2 = await client.post(
                        f"{settings.vllm_url}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.vllm_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": content},
                                {"role": "tool", "name": function_name, "content": str(execution_result)},
                                {"role": "user", "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries, no offers of help, no additional information."}
                            ],
                            "tools": tools,
                            "temperature": 0.3,
                            "max_tokens": 50
                        }
                    )
                    response2.raise_for_status()
                    result2 = response2.json()
                    return result2["choices"][0]["message"]["content"]
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse XML tool call JSON: {e}")
            
            # Debug logging
            logger.info(f"Response has tool_calls: {bool(tool_calls)}, function_call: {bool(function_call)}, content: {content[:100]}")
            if tool_calls:
                logger.info(f"Found tool_calls: {tool_calls}")
            if function_call:
                logger.info(f"Found function_call: {function_call}")
            
            if tool_calls and len(tool_calls) > 0:
                # New format: tool_calls
                tool_call = tool_calls[0]  # Get first tool call
                function_name = tool_call["function"]["name"]
                function_args_str = tool_call["function"]["arguments"]
                if isinstance(function_args_str, str):
                    function_args = json.loads(function_args_str)
                else:
                    function_args = function_args_str
                
                logger.info(f"Executing function: {function_name} with args: {function_args}")
                
                # Execute the function (pass original message for semantic matching)
                execution_result = await _execute_ha_function(function_name, function_args, original_message=message)
                
                logger.info(f"Function execution result: {execution_result}")
                
                # Check if execution result is an error - if so, return it directly
                if isinstance(execution_result, str) and (execution_result.startswith("Error:") or execution_result.startswith("Unknown function:")):
                    logger.warning(f"Function execution returned error: {execution_result}")
                    return execution_result
                
                # Second call: Get natural language response about the action
                response2 = await client.post(
                    f"{settings.vllm_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.vllm_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": None, "tool_calls": tool_calls},
                            {"role": "tool", "name": function_name, "content": str(execution_result), "tool_call_id": tool_call["id"]},
                            {"role": "user", "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries, no offers of help, no additional information."}
                        ],
                        "tools": tools,
                        "temperature": 0.3,
                        "max_tokens": 50
                    }
                )
                response2.raise_for_status()
                result2 = response2.json()
                return result2["choices"][0]["message"]["content"]
            
            elif function_call:
                # Old format: function_call (for backwards compatibility)
                function_name = function_call["name"]
                function_args_str = function_call["arguments"]
                if isinstance(function_args_str, str):
                    function_args = json.loads(function_args_str)
                else:
                    function_args = function_args_str
                
                logger.info(f"Executing function: {function_name} with args: {function_args}")
                
                # Execute the function (pass original message for semantic matching)
                execution_result = await _execute_ha_function(function_name, function_args, original_message=message)
                
                logger.info(f"Function execution result: {execution_result}")
                
                # Check if execution result is an error - if so, return it directly
                if isinstance(execution_result, str) and (execution_result.startswith("Error:") or execution_result.startswith("Unknown function:")):
                    logger.warning(f"Function execution returned error: {execution_result}")
                    return execution_result
                
                # Second call: Get natural language response about the action
                response2 = await client.post(
                    f"{settings.vllm_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.vllm_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": None, "function_call": function_call},
                            {"role": "function", "name": function_name, "content": str(execution_result)},
                            {"role": "user", "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries, no offers of help, no additional information."}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 50
                    }
                )
                response2.raise_for_status()
                result2 = response2.json()
                return result2["choices"][0]["message"]["content"]
            else:
                # No function call - check if we should auto-detect and call a function
                content = message_obj.get("content", "")
                
                # Detect if LLM is describing an action instead of taking it
                action_descriptors = [
                    "i need to", "i'll", "i will", "let me", "let's", "i should",
                    "to provide", "i can", "i would", "we need to", "we should"
                ]
                is_describing_action = any(desc in content.lower() for desc in action_descriptors)
                
                # Auto-detect "what X do I have" patterns and call list_available_entities
                message_lower = message.lower()
                what_patterns = [
                    (r"what\s+(\w+)\s+do\s+i\s+have", None),  # "what locks do I have"
                    (r"what\s+(\w+)\s+are\s+available", None),  # "what locks are available"
                    (r"list\s+my\s+(\w+)", None),  # "list my locks"
                    (r"show\s+me\s+my\s+(\w+)", None),  # "show me my locks"
                ]
                
                import re
                detected_domain = None
                for pattern, _ in what_patterns:
                    match = re.search(pattern, message_lower)
                    if match:
                        entity_type = match.group(1)
                        # Map common entity types to domains
                        domain_map = {
                            "lock": "lock",
                            "locks": "lock",
                            "light": "light",
                            "lights": "light",
                            "switch": "switch",
                            "switches": "switch",
                            "speaker": "media_player",
                            "speakers": "media_player",
                            "media": "media_player",
                            "player": "media_player",
                            "device": None,  # No domain filter
                            "devices": None,
                            "entity": None,
                            "entities": None,
                        }
                        detected_domain = domain_map.get(entity_type)
                        if detected_domain or entity_type in ["device", "devices", "entity", "entities"]:
                            logger.info(f"Auto-detected 'what {entity_type} do I have' pattern, calling list_available_entities")
                            # Call the function directly
                            function_args = {}
                            if detected_domain:
                                function_args["domain"] = detected_domain
                            execution_result = await _execute_ha_function("list_available_entities", function_args)
                            
                            # Return the result directly (no need for second LLM call for list queries)
                            return execution_result
                
                # If LLM is describing an action instead of taking it, force a retry with explicit instruction
                if is_describing_action and not detected_domain:
                    logger.warning(f"LLM described action instead of calling function. Content: {content[:100]}")
                    logger.info("Retrying with explicit function call instruction")
                    
                    # Retry with explicit instruction
                    retry_response = await client.post(
                        f"{settings.vllm_url}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.vllm_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": "You must call a function to answer this question. Do not describe what you will do - actually call the function now."}
                            ],
                            "tools": tools,
                            "tool_choice": "required",  # Force a tool call
                            "temperature": 0.3,
                            "max_tokens": 500
                        }
                    )
                    retry_response.raise_for_status()
                    retry_result = retry_response.json()
                    retry_message_obj = retry_result["choices"][0]["message"]
                    
                    # Check if retry got a function call
                    retry_tool_calls = retry_message_obj.get("tool_calls")
                    retry_function_call = retry_message_obj.get("function_call")
                    
                    if retry_tool_calls or retry_function_call:
                        # Process the function call (reuse the logic above)
                        # For simplicity, we'll handle tool_calls format
                        if retry_tool_calls and len(retry_tool_calls) > 0:
                            tool_call = retry_tool_calls[0]
                            function_name = tool_call["function"]["name"]
                            function_args_str = tool_call["function"]["arguments"]
                            if isinstance(function_args_str, str):
                                function_args = json.loads(function_args_str)
                            else:
                                function_args = function_args_str
                            
                            logger.info(f"Retry: Executing function: {function_name} with args: {function_args}")
                            execution_result = await _execute_ha_function(function_name, function_args, original_message=message)
                            logger.info(f"Retry: Function execution result: {execution_result}")
                            
                            # Check if execution result is an error - if so, return it directly
                            if isinstance(execution_result, str) and (execution_result.startswith("Error:") or execution_result.startswith("Unknown function:")):
                                logger.warning(f"Retry function execution returned error: {execution_result}")
                                return execution_result
                            
                            # For list queries, return directly; for actions, get natural language response
                            if function_name == "list_available_entities":
                                return execution_result
                            else:
                                # Get natural language response
                                response2 = await client.post(
                                    f"{settings.vllm_url}/v1/chat/completions",
                                    headers={
                                        "Authorization": f"Bearer {settings.vllm_api_key}",
                                        "Content-Type": "application/json"
                                    },
                                    json={
                                        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                                        "messages": [
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": message},
                                            {"role": "assistant", "content": None, "tool_calls": retry_tool_calls},
                                            {"role": "tool", "name": function_name, "content": str(execution_result), "tool_call_id": tool_call["id"]},
                                            {"role": "user", "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries, no offers of help, no additional information."}
                                        ],
                                        "tools": tools,
                                        "temperature": 0.3,
                                        "max_tokens": 100
                                    }
                                )
                                response2.raise_for_status()
                                result2 = response2.json()
                                return result2["choices"][0]["message"]["content"]
                
                # No function call and no auto-detection - return the response
                logger.warning(f"No function call detected. Returning LLM response: {content[:100]}")
                return content
    
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
            if function_name == "play_media_on_speakers":
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
                    # Map entity names to IDs (simplified - would need proper mapping)
                    entity_ids = entity_names
                
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
                result = await ha_client.call_service(
                    "shopping_list",
                    "add_item",
                    name=item_name
                )
                return f"Added {item_name} to shopping list"
            
            elif function_name == "start_timer":
                duration_minutes = args.get("duration_minutes", 0)
                name = args.get("name")
                duration_seconds = duration_minutes * 60
                service_data = {"duration": duration_seconds}
                if name:
                    service_data["name"] = name
                
                result = await ha_client.call_service(
                    "timer",
                    "start",
                    **service_data
                )
                timer_name = name or "timer"
                return f"Started {timer_name} for {duration_minutes} minutes"
            
            elif function_name == "get_entity_state":
                entity_id = args.get("entity_id", "")
                entity = await ha_client.get_entity(entity_id)
                return f"Entity {entity_id} state: {entity.get('state', 'unknown')}"
            
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
                    if likely_domain:
                        logger.info(f"Attempting semantic matching for '{search_term}' in domain '{likely_domain}'")
                        try:
                            # Get all entities in the domain
                            all_entities = await ha_client.get_entities()
                            domain_entities = [e for e in all_entities if e["entity_id"].startswith(f"{likely_domain}.")]
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
                                
                                async with httpx.AsyncClient(timeout=10.0) as llm_client:
                                    match_response = await llm_client.post(
                                        f"{settings.vllm_url}/v1/chat/completions",
                                        headers={
                                            "Authorization": f"Bearer {settings.vllm_api_key}",
                                            "Content-Type": "application/json"
                                        },
                                        json={
                                            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                                            "messages": [
                                                {"role": "system", "content": "You match user descriptions to Home Assistant entities using semantic understanding. Match based on meaning: vehicle names/brands match 'car', room names match locations, device types match functions. Return ONLY the exact entity_id that best matches, nothing else."},
                                                {"role": "user", "content": f"User said: '{original_message if original_message else search_term}'\n\nAvailable {likely_domain} entities:\n{entity_list_str}\n\nWhich entity_id semantically matches what the user wants to control? Use semantic understanding - for example, 'car' matches vehicle names like 'rav4', 'toyota', etc. Return only the entity_id."}
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
                                    import re
                                    entity_id_match = re.search(r'([a-z_]+\.\S+)', matched_entity_id)
                                    if entity_id_match:
                                        matched_entity_id = entity_id_match.group(1)
                                    
                                    # Verify the matched entity exists
                                    candidate_ids = [e.get("entity_id", "") for e in domain_entities]
                                    if matched_entity_id in candidate_ids:
                                        entity_id = matched_entity_id
                                        domain = likely_domain
                                        logger.info(f"Semantically matched '{search_term}' to entity_id '{entity_id}' in domain '{likely_domain}'")
                                        resolved = True
                                    else:
                                        logger.warning(f"LLM returned entity_id not in candidates: {matched_entity_id}. Candidates: {candidate_ids[:3]}")
                                        # If semantic matching returned invalid entity, don't fall back to literal search
                                        # Return error instead
                                        error_msg = f"Error: Semantic matching failed to find valid {likely_domain} entity matching '{original_message if original_message else search_term}'"
                                        logger.warning(error_msg)
                                        return error_msg
                        except Exception as e:
                            logger.error(f"Error in semantic entity matching: {e}", exc_info=True)
                            # If semantic matching fails with a known domain, return error instead of falling back
                            error_msg = f"Error: Could not semantically match '{original_message if original_message else search_term}' to a {likely_domain} entity: {str(e)}"
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
                                            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
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
                                    import re
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


async def _handle_googling(message: str) -> str:
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
        try:
            with open("/opt/llm/prompts/googling_system.txt", "r") as f:
                system_prompt = f.read()
        except Exception:
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
                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {message}\n\nSearch Results:\n{search_context}\n\nAnswer the question concisely:"}
                    ],
                    "temperature": 0.3,  # Lower temperature for factual answers
                    "max_tokens": 300
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
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


async def _handle_general_chat(message: str) -> str:
    """
    Handle general chat requests using vLLM
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                    "messages": [
                        {"role": "user", "content": message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
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
