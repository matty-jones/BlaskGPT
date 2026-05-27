"""
HA command execution with sub-use-case prompts and scoped tools.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ha_router import (
    HASubUseCase,
    build_system_prompt,
    classify_ha_sub_use_case,
    get_tools_for_sub_use_case,
)
from tools.ha_functions import get_function_schemas, get_function_schemas_by_names
from music_matcher import (
    extract_type_hint,
    extract_artist_hint,
    extract_live_hint,
    select_best_match,
)

logger = logging.getLogger(__name__)


def _minutes_to_hhmmss(minutes: int) -> str:
    """Convert minutes to HH:MM:SS format for Home Assistant timer scripts."""
    total_seconds = minutes * 60
    hours = total_seconds // 3600
    minutes_remaining = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes_remaining:02d}:{seconds:02d}"


ACTION_FUNCTIONS = {
    "play_media_on_speakers",
    "stop_media_on_speakers",
    "control_media_playback",
    "set_volume_on_speakers",
    "add_item_to_shopping_list",
    "start_timer",
    "turn_on_entity",
    "turn_off_entity",
    "find_phone_blasxel_6",
}
LISTING_FUNCTIONS = {
    "list_available_entities",
    "get_shopping_list_items",
    "get_entity_state",
}

# Simple in-memory cache of entity listings by domain (strings as returned to LLM)
_ENTITY_CACHE: Dict[str, str] = {}
_CACHE_LOCK = asyncio.Lock()
_REFRESH_TASK: Optional[asyncio.Task] = None
_REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # daily

# Context-aware speaker tracking: stores last interacted speakers per user (or globally)
# Format: {user_id: [entity_id1, entity_id2, ...]} or {"default": [...]} for global context
_LAST_INTERACTED_SPEAKERS: Dict[str, List[str]] = {}
_SPEAKER_CONTEXT_LOCK = asyncio.Lock()


def _get_user_context_key(user_id: Optional[str] = None) -> str:
    """Get context key for user (or 'default' if no user ID)"""
    return user_id if user_id else "default"


async def _track_speakers(entity_ids: List[str], user_id: Optional[str] = None):
    """Track speakers that were just interacted with"""
    async with _SPEAKER_CONTEXT_LOCK:
        key = _get_user_context_key(user_id)
        _LAST_INTERACTED_SPEAKERS[key] = entity_ids.copy()
        logger.info(f"[Context] Tracked speakers for user '{key}': {entity_ids}")


async def _get_last_speakers(user_id: Optional[str] = None) -> List[str]:
    """Get last interacted speakers for user (or empty list if none)"""
    async with _SPEAKER_CONTEXT_LOCK:
        key = _get_user_context_key(user_id)
        return _LAST_INTERACTED_SPEAKERS.get(key, []).copy()


async def handle_ha_command(message: str, ha_client, settings, max_tokens: int = 200) -> str:
    """Route to sub-use-case and execute with scoped tools/prompt."""
    sub_use_case = classify_ha_sub_use_case(message)
    system_prompt = build_system_prompt(sub_use_case)
    tool_names = get_tools_for_sub_use_case(sub_use_case)
    tools = get_function_schemas_by_names(tool_names) if tool_names else []
    if not tools:
        tools = get_function_schemas()

    # Pre-load entity listings for lights/switches and media players to avoid first-turn guesses
    extra_tool_messages: List[Dict[str, Any]] = []
    if sub_use_case == HASubUseCase.LIGHTS_SWITCHES_LOCKS:
        extra_tool_messages = await _prefetch_light_switch_listings(ha_client)
    elif sub_use_case == HASubUseCase.MUSIC:
        extra_tool_messages = await _prefetch_media_player_listings(ha_client)

    return await _run_tool_flow(
        message=message,
        system_prompt=system_prompt,
        tools=tools,
        ha_client=ha_client,
        settings=settings,
        extra_tool_messages=extra_tool_messages,
        max_tokens=max_tokens,
    )


async def _run_tool_flow(
    message: str,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    ha_client,
    settings,
    extra_tool_messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 200,
) -> str:
    """Single tool-call turn followed by brief confirmation or list formatting."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        first = await client.post(
            f"{settings.vllm_url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.vllm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "blaskgpt",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                    *(
                        extra_tool_messages
                        if extra_tool_messages
                        else []
                    ),
                ],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
            },
        )
        if first.status_code != 200:
            try:
                error_text = first.text
                logger.error(f"vLLM returned {first.status_code}: {error_text[:500]}")
            except:
                logger.error(f"vLLM returned {first.status_code} but could not read error body")
            # Log a sample of tools for debugging
            try:
                tools_sample = json.dumps(tools[:1] if tools else [], indent=2)[:500] if tools else "No tools"
                logger.error(f"Sample tool schema: {tools_sample}")
            except:
                pass
        first.raise_for_status()
        first_result = first.json()
        message_obj = first_result["choices"][0]["message"]
        tool_calls = message_obj.get("tool_calls")
        content = message_obj.get("content", "")

        # Handle XML-style tool call content from some model parsers
        if not tool_calls and isinstance(content, str):
            xml_match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", content, re.DOTALL)
            if xml_match:
                try:
                    tool_call_data = json.loads(xml_match.group(1))
                    function_name = tool_call_data.get("name")
                    function_args = tool_call_data.get("arguments", {})
                    tool_calls = [
                        {
                            "id": "xml_tool_call",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(function_args),
                            },
                        }
                    ]
                    logger.info(f"Parsed XML-style tool call: {function_name} with args {function_args}")
                except Exception as e:
                    logger.warning(f"Failed to parse XML-style tool call: {e}")

        if tool_calls:
            tool_call = tool_calls[0]
            function_name = tool_call["function"]["name"]
            args_raw = tool_call["function"]["arguments"]
            function_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

            logger.info(f"Executing HA tool: {function_name} with args: {function_args}")
            if function_name == "play_media_on_speakers":
                logger.info(f"[MA-debug] ====== play_media_on_speakers CALLED ======")
            execution_result = await _execute_tool_call(
                function_name=function_name,
                function_args=function_args,
                user_message=message,
                ha_client=ha_client,
            )

            # If the model guessed a bad entity_id for on/off or speakers, retry by listing entities and forcing a new tool call
            if (
                isinstance(execution_result, str)
                and execution_result.lower().startswith("error")
                and function_name in ("turn_on_entity", "turn_off_entity", "play_media_on_speakers")
                and ("entity" in execution_result.lower() or "speaker" in execution_result.lower())
                and ("not found" in execution_result.lower() or "must first call" in execution_result.lower())
            ):
                retry_resp = await _retry_with_entity_listing(
                    client=client,
                    system_prompt=system_prompt,
                    tools=tools,
                    user_message=message,
                    prior_tool_calls=tool_calls,
                    prior_tool_name=function_name,
                    prior_tool_result=execution_result,
                    ha_client=ha_client,
                    settings=settings,
                    max_tokens=max_tokens,
                )
                if retry_resp:
                    return retry_resp
                # If retry failed to produce a response, fall through and return the original error

            if isinstance(execution_result, str) and execution_result.startswith("Error"):
                logger.warning(f"Tool execution returned error: {execution_result}")
                return execution_result

            # Fast path: simple on/off actions do not need a second LLM call just to say "done".
            if function_name in ("turn_on_entity", "turn_off_entity"):
                entity_id = function_args.get("entity_id", "entity")
                state_word = "on" if function_name == "turn_on_entity" else "off"
                friendly = entity_id.replace("switch.", "").replace("light.", "").replace("_", " ")
                return f"{friendly.title()} turned {state_word}."

            follow_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "name": function_name,
                    "content": str(execution_result),
                    "tool_call_id": tool_call["id"],
                },
            ]

            # Check if LLM wants to make another tool call (e.g., play_media_on_speakers after search_music)
            second = await client.post(
                f"{settings.vllm_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "blaskgpt",
                    "messages": follow_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    },
                },
            )
            second.raise_for_status()
            second_result = second.json()
            second_message = second_result["choices"][0]["message"]
            second_tool_calls = second_message.get("tool_calls")
            second_content = second_message.get("content", "")

            # Handle XML-style tool call content from some model parsers
            if not second_tool_calls and isinstance(second_content, str):
                xml_match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", second_content, re.DOTALL)
                if xml_match:
                    try:
                        tool_call_data = json.loads(xml_match.group(1))
                        function_name_2 = tool_call_data.get("name")
                        function_args_2 = tool_call_data.get("arguments", {})
                        second_tool_calls = [
                            {
                                "id": "xml_tool_call_2",
                                "type": "function",
                                "function": {
                                    "name": function_name_2,
                                    "arguments": json.dumps(function_args_2),
                                },
                            }
                        ]
                        logger.info(f"Parsed XML-style tool call (second): {function_name_2} with args {function_args_2}")
                    except Exception as e:
                        logger.warning(f"Failed to parse XML-style tool call (second): {e}")

            # If there's a second tool call, execute it
            if second_tool_calls:
                second_tool_call = second_tool_calls[0]
                second_function_name = second_tool_call["function"]["name"]
                second_args_raw = second_tool_call["function"]["arguments"]
                second_function_args = json.loads(second_args_raw) if isinstance(second_args_raw, str) else second_args_raw

                logger.info(f"Executing second HA tool: {second_function_name} with args: {second_function_args}")
                second_execution_result = await _execute_tool_call(
                    function_name=second_function_name,
                    function_args=second_function_args,
                    user_message=message,
                    ha_client=ha_client,
                )

                # Final response after second tool call
                final_messages = follow_messages + [
                    {"role": "assistant", "content": None, "tool_calls": second_tool_calls},
                    {
                        "role": "tool",
                        "name": second_function_name,
                        "content": str(second_execution_result),
                        "tool_call_id": second_tool_call["id"],
                    },
                    {
                        "role": "user",
                        "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries or extra information.",
                    },
                ]

                final = await client.post(
                    f"{settings.vllm_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.vllm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "blaskgpt",
                        "messages": final_messages,
                        "tools": tools,
                        "temperature": 0.3,
                        "max_tokens": max_tokens,
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        },
                    },
                )
                final.raise_for_status()
                final_result = final.json()
                return final_result["choices"][0]["message"]["content"]

            # No second tool call; return the response
            if function_name in LISTING_FUNCTIONS:
                return second_content
            else:
                # For action functions, if no second tool call, ask for confirmation
                if not second_content:
                    return "Done."
                return second_content

        # No tool call; return model content
        return content


async def _retry_with_entity_listing(
    client: httpx.AsyncClient,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    user_message: str,
    prior_tool_calls: List[Dict[str, Any]],
    prior_tool_name: str,
    prior_tool_result: str,
    ha_client,
    settings,
    max_tokens: int = 200,
) -> Optional[str]:
    """When entity_id was guessed and not found, list entities and force a new tool call."""
    listings_parts: List[str] = []
    # Determine which domains to list based on the function that failed
    if prior_tool_name == "play_media_on_speakers":
        domains = ["media_player"]
    else:
        domains = ["light", "switch"]
    
    for domain in domains:
        listing = await _get_or_refresh_listing(domain, ha_client)
        if listing:
            listings_parts.append(f"{domain} domain:\n{listing}")
    if not listings_parts:
        return None

    combined_listing = "\n\n".join(listings_parts)
    retry_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": None, "tool_calls": prior_tool_calls},
        {
            "role": "tool",
            "name": prior_tool_name,
            "content": str(prior_tool_result),
            "tool_call_id": prior_tool_calls[0].get("id", "invalid_entity"),
        },
        {
            "role": "tool",
            "name": "list_available_entities",
            "content": combined_listing,
            "tool_call_id": "entity_listing",
        },
        {
            "role": "user",
            "content": (
                "From the lists above, choose the best matching entity for the user's request "
                f"and call {prior_tool_name} with the exact entity_id. Do not guess. "
                "Call a tool now."
            ),
        },
    ]

    retry = await client.post(
        f"{settings.vllm_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.vllm_api_key}",
            "Content-Type": "application/json",
        },
            json={
                "model": "blaskgpt",
                "messages": retry_messages,
                "tools": tools,
                "tool_choice": "required",
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
            },
    )
    retry.raise_for_status()
    retry_result = retry.json()
    retry_msg = retry_result["choices"][0]["message"]
    retry_calls = retry_msg.get("tool_calls")
    if not retry_calls:
        return None

    retry_call = retry_calls[0]
    retry_fn = retry_call["function"]["name"]
    retry_args_raw = retry_call["function"]["arguments"]
    retry_args = json.loads(retry_args_raw) if isinstance(retry_args_raw, str) else retry_args_raw

    logger.info(f"Retry tool call selected: {retry_fn} with args {retry_args}")
    execution_result2 = await _execute_tool_call(
        function_name=retry_fn,
        function_args=retry_args,
        user_message=user_message,
        ha_client=ha_client,
    )

    if isinstance(execution_result2, str) and execution_result2.startswith("Error"):
        return execution_result2

    follow_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": None, "tool_calls": retry_calls},
        {
            "role": "tool",
            "name": retry_fn,
            "content": str(execution_result2),
            "tool_call_id": retry_call.get("id", "retry_call"),
        },
        {
            "role": "user",
            "content": "Respond with only a brief confirmation. Maximum 10 words. No pleasantries or extra information.",
        },
    ]

    final = await client.post(
        f"{settings.vllm_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.vllm_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "blaskgpt",
            "messages": follow_messages,
            "tools": tools,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        },
    )
    final.raise_for_status()
    final_result = final.json()
    return final_result["choices"][0]["message"]["content"]


async def _execute_tool_call(
    function_name: str,
    function_args: Dict[str, Any],
    user_message: str,
    ha_client,
) -> Any:
    """Execute a tool with small domain-specific fallbacks."""
    # Special case: lights may live under switch
    if function_name == "list_available_entities" and function_args.get("domain") == "light":
        execution_result = await _execute_ha_function(
            ha_client=ha_client,
            function_name=function_name,
            args=function_args,
            original_message=user_message,
        )
        if "Found 0 entities" in str(execution_result):
            switch_result = await _execute_ha_function(
                ha_client=ha_client,
                function_name="list_available_entities",
                args={"domain": "switch"},
                original_message=user_message,
            )
            if "Found 0 entities" not in str(switch_result):
                return f"Found in 'light' domain: {execution_result}\n\nFound in 'switch' domain (may include light switches): {switch_result}"
        return execution_result

    return await _execute_ha_function(
        ha_client=ha_client,
        function_name=function_name,
        args=function_args,
        original_message=user_message,
    )


def _filter_dummy_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out dummy/helper entities that should not be used for control."""
    filtered = []
    for entity in entities:
        entity_id = entity.get("entity_id", "").lower()
        friendly_name = entity.get("attributes", {}).get("friendly_name", "").lower()
        entity_id_original = entity.get("entity_id", "")

        # Filter out dummy entities and group/helper entities
        is_dummy = "dummy" in entity_id or "dummy" in friendly_name
        is_group_helper = (
            "all_the_speakers" in entity_id or
            "all_speakers" in entity_id or
            entity_id.endswith("_all") or
            entity_id.startswith("all_")
        )

        if not is_dummy and not is_group_helper:
            filtered.append(entity)
        else:
            filter_reason = "dummy" if is_dummy else "group/helper"
            logger.debug(f"Filtered out {filter_reason} entity: {entity_id_original}")

    return filtered


async def _is_music_assistant_entity(ha_client, entity_id: str) -> bool:
    """Check if an entity is provided by Music Assistant integration."""
    try:
        # Try different entity registry endpoints
        entity_registry = None
        for endpoint in ["config/entity_registry/list", "config/entity_registry", "entity_registry/list"]:
            try:
                entity_registry = await ha_client._request("GET", endpoint)
                if entity_registry:
                    break
            except Exception:
                continue
        
        if isinstance(entity_registry, list):
            for entry in entity_registry:
                if entry.get("entity_id") == entity_id:
                    platform = entry.get("platform", "").lower()
                    config_entry_id = entry.get("config_entry_id", "")
                    # Music Assistant entities typically have platform "music_assistant" or the MA config_entry_id
                    if "music_assistant" in platform or config_entry_id == "01K194WFCH402CXEG8B2VZ620E":
                        logger.info(f"[MA-debug] Entity {entity_id} identified as Music Assistant via registry (platform={platform}, config_entry_id={config_entry_id})")
                        return True
        
        # Fallback: check if entity_id ends with _2 (common pattern for Music Assistant duplicates)
        # This is a heuristic - Music Assistant often creates entities with _2 suffix
        if entity_id.endswith("_2") and "media_player" in entity_id:
            logger.info(f"[MA-debug] Entity {entity_id} identified as Music Assistant via _2 suffix heuristic")
            return True
        return False
    except Exception as e:
        logger.debug(f"Could not check if {entity_id} is Music Assistant entity: {e}")
        # Fallback heuristic
        if entity_id.endswith("_2") and "media_player" in entity_id:
            logger.info(f"[MA-debug] Entity {entity_id} identified as Music Assistant via _2 suffix heuristic (fallback)")
            return True
        return False


async def _execute_ha_function(
    ha_client,
    function_name: str,
    args: Dict[str, Any],
    original_message: Optional[str] = None,
) -> Any:
    """Execute a Home Assistant function."""
    try:
        async with ha_client:
            if function_name == "search_music":
                query = args.get("query", "")
                raw_utterance = args.get("raw_utterance", "")
                area = args.get("area", "")
                media_type_hint = args.get("media_type", "any")
                limit = args.get("limit", 10)

                if not query:
                    return "Error: No search query provided"
                
                # Extract hints from raw utterance
                parsed_type_hint = extract_type_hint(raw_utterance) if raw_utterance else None
                artist_hint = extract_artist_hint(raw_utterance) if raw_utterance else None
                live_hint = extract_live_hint(raw_utterance) if raw_utterance else False
                
                # Combine type hints: parsed hint takes precedence, but LLM hint is also considered
                # If parsed hint exists, use it; otherwise use LLM hint if not "any"
                final_type_hint = parsed_type_hint if parsed_type_hint else (media_type_hint if media_type_hint != "any" else None)
                
                logger.info(f"[MA-debug] Query: '{query}', parsed_type_hint: {parsed_type_hint}, LLM_type_hint: {media_type_hint}, final_type_hint: {final_type_hint}, artist_hint: {artist_hint}, live_hint: {live_hint}")

                try:
                    speakers = await ha_client.find_speakers()
                    speakers = _filter_dummy_entities(speakers)

                    if not speakers:
                        return "Error: No media player entities found. Please ensure Music Assistant is configured."

                    # Try to get config_entry_id from config entries directly (Music Assistant integration)
                    config_entry_id = None
                    try:
                        # Get all config entries and find Music Assistant
                        config_entries = await ha_client._request("GET", "config/config_entries/entry")
                        logger.info(f"[MA-debug] Found {len(config_entries) if isinstance(config_entries, list) else 'unknown'} config entries")
                        
                        if isinstance(config_entries, list):
                            for entry in config_entries:
                                domain = entry.get("domain", "")
                                entry_id = entry.get("entry_id", "")
                                title = entry.get("title", "")
                                if domain == "music_assistant":
                                    config_entry_id = entry_id
                                    logger.info(f"[MA-debug] Found Music Assistant config_entry_id: {config_entry_id}")
                                    break
                    except Exception as e:
                        logger.warning(f"[MA-debug] Could not get config entries directly: {e}")
                    
                    # Fallback: Try device registry approach if config entries didn't work
                    if not config_entry_id:
                        logger.info("[MA-debug] Trying device registry approach as fallback...")
                        for speaker in speakers:
                            ma_entity = speaker["entity_id"]
                            friendly = speaker.get("attributes", {}).get("friendly_name", "")
                            logger.info(f"[MA-debug] Speaker candidate: entity_id={ma_entity}, friendly_name={friendly}")
                            try:
                                entity_state = await ha_client.get_entity(ma_entity)
                                device_id = entity_state.get("attributes", {}).get("device_id")
                                logger.info(f"[MA-debug] Entity state for {ma_entity}: device_id={device_id}")

                                if device_id:
                                    try:
                                        device_info = await ha_client._request("GET", f"config/device_registry/{device_id}")
                                        config_entries = device_info.get("config_entries", [])
                                        identifiers = device_info.get("identifiers", [])
                                        logger.info(
                                            f"[MA-debug] Device info for {ma_entity}: config_entries={config_entries}, identifiers={identifiers}"
                                        )
                                        if config_entries:
                                            config_entry_id = config_entries[0]
                                            logger.info(f"Found config_entry_id: {config_entry_id} for entity {ma_entity}")
                                            break  # Found a valid one, stop searching
                                    except Exception as e:
                                        logger.debug(f"Could not get config_entry_id from device registry for {ma_entity}: {e}")
                                        continue
                            except Exception as e:
                                logger.debug(f"Could not get device_id from entity {ma_entity}: {e}")
                                continue

                    search_data = {
                        "name": query
                    }

                    # Prefer config_entry_id when found, but if none found, try without it as a fallback.
                    if config_entry_id:
                        search_data["config_entry_id"] = config_entry_id
                    else:
                        logger.warning(
                            "[MA-debug] No config_entry_id found from media players; calling music_assistant.search without it"
                        )

                    # Only pass media_type to search if we have a strong hint (not "any")
                    # The fuzzy matcher will handle type preference, but we can help narrow search
                    if final_type_hint:
                        search_data["media_type"] = final_type_hint
                    elif media_type_hint and media_type_hint != "any":
                        search_data["media_type"] = media_type_hint
                    
                    if limit:
                        search_data["limit"] = limit

                    logger.info(f"[MA-debug] Calling music_assistant.search with data: {search_data}")
                    result = await ha_client.call_service(
                        "music_assistant",
                        "search",
                        return_response=True,
                        **search_data
                    )
                    logger.info(f"[MA-debug] Raw search result type: {type(result)}, content: {str(result)[:500]}")

                    items = []
                    if isinstance(result, dict):
                        service_response = result.get("service_response", {})
                        logger.info(f"[MA-debug] service_response type: {type(service_response)}, keys: {service_response.keys() if isinstance(service_response, dict) else 'N/A'}")
                        if isinstance(service_response, dict):
                            # Music Assistant returns results in separate keys: tracks, artists, albums, playlists, etc.
                            # Combine all result types based on what's available
                            all_results = []
                            for key in ["tracks", "artists", "albums", "playlists", "radio", "audiobooks", "podcasts"]:
                                if key in service_response and isinstance(service_response[key], list):
                                    all_results.extend(service_response[key])
                            items = all_results
                            # Fallback to old format if new format doesn't have results
                            if not items:
                                items = service_response.get("items", service_response.get("results", []))
                        elif isinstance(service_response, list):
                            items = service_response
                    elif isinstance(result, list):
                        items = result

                    logger.info(f"[MA-debug] Extracted {len(items)} items from search result")
                    if not items:
                        logger.warning(f"[MA-debug] No items found in search result for query: {query}")
                        return f"No music found for query: {query}. Try a different search term or check that Music Assistant has music available."

                    logger.info(f"[MA-debug] Search returned {len(items)} items for query '{query}'")
                    
                    # Use fuzzy matching to select best match
                    best_match = select_best_match(
                        query=query,
                        search_results=items[:limit],  # Limit items for scoring
                        type_hint=final_type_hint,
                        artist_hint=artist_hint,
                        live_hint=live_hint,
                    )
                    
                    if not best_match:
                        return f"No strong match found for '{query}'. Please be more specific (e.g., include artist name, specify 'playlist' or 'album')."
                    
                    logger.info(f"[MA-debug] Selected best match: {best_match['name']} ({best_match['media_type']}) - URI: {best_match['uri']}")
                    
                    # Determine speakers to play on
                    entity_ids = []
                    if area:
                        # Try to resolve area to entity_id
                        resolved_entities = await ha_client.search_entities(area, domain="media_player")
                        if resolved_entities:
                            # For Music Assistant library URIs, prefer _2 entities (Music Assistant duplicates)
                            if best_match["uri"].startswith("library://"):
                                # Filter to prefer Music Assistant entities (_2 suffix)
                                ma_entities = [e for e in resolved_entities if e["entity_id"].endswith("_2")]
                                if ma_entities:
                                    entity_ids = [e["entity_id"] for e in ma_entities]
                                    logger.info(f"[MA-debug] Resolved area '{area}' to Music Assistant entities: {entity_ids}")
                                else:
                                    # Try converting non-_2 entities to _2
                                    for e in resolved_entities:
                                        eid = e["entity_id"]
                                        if not eid.endswith("_2"):
                                            ma_candidate = eid + "_2"
                                            try:
                                                await ha_client.get_entity(ma_candidate)
                                                entity_ids.append(ma_candidate)
                                                logger.info(f"[MA-debug] Found Music Assistant entity: {ma_candidate} (from {eid})")
                                            except Exception:
                                                # Fallback to original if _2 doesn't exist
                                                entity_ids.append(eid)
                                    if entity_ids:
                                        logger.info(f"[MA-debug] Resolved area '{area}' to entities (converted to MA): {entity_ids}")
                            else:
                                entity_ids = [e["entity_id"] for e in resolved_entities]
                                logger.info(f"[MA-debug] Resolved area '{area}' to entities: {entity_ids}")
                    
                    # If no area specified or resolution failed, use context-aware last speakers
                    if not entity_ids:
                        entity_ids = await _get_last_speakers()
                        if entity_ids:
                            logger.info(f"[MA-debug] Using context-aware last speakers: {entity_ids}")
                    
                    # If still no speakers, ask user to specify
                    if not entity_ids:
                        return f"Found match: {best_match['name']} ({best_match['media_type']}), but no speakers specified. Please specify which speakers to use (e.g., 'on Den Wifi' or 'on the living room speaker')."
                    
                    # Deduplicate entity_ids
                    entity_ids = list(dict.fromkeys(entity_ids))  # Preserves order while removing duplicates
                    
                    # Determine media_content_type based on selected match type
                    media_content_type_map = {
                        "track": "music",
                        "artist": "music",
                        "playlist": "playlist",
                        "album": "album",
                    }
                    media_content_type = media_content_type_map.get(best_match["media_type"], "music")
                    
                    # Automatically play the selected match
                    logger.info(f"[MA-debug] Auto-playing {best_match['name']} ({best_match['media_type']}) on {entity_ids}")
                    results = []
                    
                    # For tracks, use the script to play track then append artist radio
                    if best_match["media_type"] == "track":
                        logger.info(f"[MA-debug] Track detected, calling script.play_track_then_artist_radio")
                        for eid in entity_ids:
                            try:
                                logger.info(f"[MA-debug] Calling script.play_track_then_artist_radio for {eid} with track_uri={best_match['uri']}, artist_name={best_match.get('artist', '')}")
                                script_result = await ha_client.call_service(
                                    "script",
                                    "turn_on",
                                    entity_id="script.play_track_then_artist_radio",
                                    variables={
                                        "player_entity": eid,
                                        "track_uri": best_match["uri"],
                                        "artist_name": best_match.get("artist", "")
                                    }
                                )
                                logger.info(f"[MA-debug] script.play_track_then_artist_radio result for {eid}: {script_result}")
                                results.append(f"Success for {eid}")
                            except Exception as e:
                                logger.error(f"[MA-debug] Script call failed for {eid}: {e}", exc_info=True)
                                error_str = str(e)
                                # If script doesn't exist, fall back to direct playback
                                if "404" in error_str or "not found" in error_str.lower() or "unavailable" in error_str.lower():
                                    logger.warning(f"[MA-debug] Script not found, falling back to direct playback for {eid}")
                                    try:
                                        set_result = await ha_client.call_service(
                                            "media_player",
                                            "play_media",
                                            entity_id=eid,
                                            media_content_id=best_match["uri"],
                                            media_content_type=media_content_type,
                                        )
                                        play_result = await ha_client.call_service(
                                            "media_player",
                                            "media_play",
                                            entity_id=eid,
                                        )
                                        logger.info(f"[MA-debug] Fallback playback result for {eid}: {play_result}")
                                        results.append(f"Success for {eid} (fallback)")
                                    except Exception as fallback_error:
                                        logger.error(f"[MA-debug] Fallback playback also failed for {eid}: {fallback_error}", exc_info=True)
                                        results.append(f"Error for {eid}: {str(fallback_error)}")
                                else:
                                    results.append(f"Error for {eid}: {str(e)}")
                    else:
                        # For non-track media types (artists, playlists, albums), use existing playback logic
                        for eid in entity_ids:
                            try:
                                # Set the media content
                                logger.info(f"[MA-debug] Setting media_content_id on '{eid}' to '{best_match['uri']}'")
                                set_result = await ha_client.call_service(
                                    "media_player",
                                    "play_media",
                                    entity_id=eid,
                                    media_content_id=best_match["uri"],
                                    media_content_type=media_content_type,
                                )
                                logger.info(f"[MA-debug] media_player.play_media result for {eid}: {set_result}")
                                
                                # For artists/playlists, Music Assistant may need a moment to queue tracks
                                if best_match["media_type"] in ["artist", "playlist"]:
                                    await asyncio.sleep(0.5)  # Brief delay for queueing
                                
                                # Start playback
                                logger.info(f"[MA-debug] Starting playback on '{eid}'")
                                play_result = await ha_client.call_service(
                                    "media_player",
                                    "media_play",
                                    entity_id=eid,
                                )
                                logger.info(f"[MA-debug] media_player.media_play result for {eid}: {play_result}")
                                
                                # Check entity state to verify playback started
                                await asyncio.sleep(0.3)  # Brief delay before checking state
                                try:
                                    entity_state = await ha_client.get_entity(eid)
                                    state = entity_state.get("state", "unknown")
                                    logger.info(f"[MA-debug] Entity {eid} state after playback: {state}")
                                    if state not in ["playing", "buffering"]:
                                        logger.warning(f"[MA-debug] Entity {eid} is not playing (state: {state})")
                                except Exception as state_check_error:
                                    logger.warning(f"[MA-debug] Could not check state for {eid}: {state_check_error}")
                                
                                results.append(f"Success for {eid}")
                            except Exception as e:
                                logger.error(f"[MA-debug] Playback failed for {eid}: {e}", exc_info=True)
                                results.append(f"Error for {eid}: {str(e)}")
                    
                    # Track speakers for context-aware future commands
                    await _track_speakers(entity_ids)
                    
                    # Format success message
                    match_name = best_match["name"]
                    if best_match.get("artist"):
                        match_name = f"{match_name} by {best_match['artist']}"
                    
                    return f"Playing {match_name} ({best_match['media_type']}) on {', '.join(entity_ids)}"

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
                logger.info(f"[MA-debug] play_media_on_speakers called with entity_names={entity_names}, media_content_id={media_content_id}, media_content_type={media_content_type}")

                if "all" in [name.lower() for name in entity_names]:
                    speakers = await ha_client.find_speakers()
                    speakers = _filter_dummy_entities(speakers)
                    entity_ids = [s["entity_id"] for s in speakers]
                else:
                    entity_ids = []
                    invalid_entities = []

                    for entity_name in entity_names:
                        if "." in entity_name:
                            try:
                                await ha_client.get_entity(entity_name)
                                entity_ids.append(entity_name)
                            except Exception as e:
                                logger.warning(f"Entity {entity_name} not found: {e}")
                                invalid_entities.append(entity_name)
                        else:
                            invalid_entities.append(entity_name)

                    if invalid_entities:
                        return f"Error: The following speaker entities were not found: {', '.join(invalid_entities)}. You must first call list_available_entities with domain='media_player' to get the exact entity_id from Home Assistant. Do not guess or construct entity_ids."

                # For Music Assistant library URIs, automatically use _2 entities (Music Assistant duplicates)
                if media_content_id.startswith("library://"):
                    logger.info(f"[MA-debug] Library URI detected, converting to Music Assistant entities...")
                    ma_entity_ids = []
                    for eid in entity_ids:
                        # If entity already ends with _2, use it (likely Music Assistant)
                        if eid.endswith("_2"):
                            logger.info(f"[MA-debug] Entity {eid} already has _2 suffix, using as-is")
                            ma_entity_ids.append(eid)
                        else:
                            # Try the _2 version (Music Assistant often creates _2 duplicates)
                            ma_candidate = eid + "_2"
                            try:
                                await ha_client.get_entity(ma_candidate)
                                logger.info(f"[MA-debug] Found Music Assistant entity: {ma_candidate} (replacing {eid})")
                                ma_entity_ids.append(ma_candidate)
                            except Exception:
                                # _2 version doesn't exist, use original but log warning
                                logger.warning(f"[MA-debug] Music Assistant entity {ma_candidate} not found, using original {eid}")
                                ma_entity_ids.append(eid)
                    
                    if ma_entity_ids:
                        logger.info(f"[MA-debug] Using Music Assistant entities: {ma_entity_ids}")
                        entity_ids = ma_entity_ids

                if not entity_ids:
                    return "Error: No valid speaker entities found. You must first call list_available_entities with domain='media_player' to get available speakers."

                # Music Assistant library URIs: use media_player.play_media then media_player.media_play
                if media_content_id.startswith("library://"):
                    logger.info(f"[MA-debug] Using media_player.play_media + media_play for Music Assistant library URI: {media_content_id}")
                    results = []
                    for eid in entity_ids:
                        try:
                            # First, set the media content
                            logger.info(f"[MA-debug] Setting media_content_id on '{eid}' to '{media_content_id}'")
                            set_result = await ha_client.call_service(
                                "media_player",
                                "play_media",
                                entity_id=eid,
                                media_content_id=media_content_id,
                                media_content_type=media_content_type or "music",
                            )
                            logger.info(f"[MA-debug] media_player.play_media result for {eid}: {set_result}")
                            
                            # Then, start playback
                            logger.info(f"[MA-debug] Starting playback on '{eid}'")
                            play_result = await ha_client.call_service(
                                "media_player",
                                "media_play",
                                entity_id=eid,
                            )
                            logger.info(f"[MA-debug] media_player.media_play result for {eid}: {play_result}")
                            results.append(f"Success for {eid}")
                        except Exception as e:
                            logger.error(f"[MA-debug] Playback failed for {eid}: {e}", exc_info=True)
                            results.append(f"Error for {eid}: {str(e)}")
                    # Track speakers for context-aware future commands
                    await _track_speakers(entity_ids)
                    return f"Playing {media_content_id} on speakers: {', '.join(entity_ids)}"

                # Fallback: generic media_player.play_media (non-library URIs)
                logger.info(
                    f"Calling media_player.play_media on {entity_ids} with media_content_id='{media_content_id}', media_content_type='{media_content_type}'"
                )
                result = await ha_client.call_service(
                    "media_player",
                    "play_media",
                    entity_id=entity_ids,
                    media_content_id=media_content_id,
                    media_content_type=media_content_type
                )
                logger.info(f"media_player.play_media service result: {result}")
                # Track speakers for context-aware future commands
                await _track_speakers(entity_ids)
                return f"Playing {media_content_id} on speakers: {', '.join(entity_ids)}"

            elif function_name == "stop_media_on_speakers":
                entity_names = args.get("entity_names", [])
                action = args.get("action", "stop")

                # Check if "all" is requested (check both the list and individual entity names)
                is_all_requested = (
                    "all" in [name.lower() for name in entity_names] or
                    any("all" in name.lower() for name in entity_names)
                )
                
                if is_all_requested:
                    speakers = await ha_client.find_speakers()
                    speakers = _filter_dummy_entities(speakers)
                    entity_ids = [s["entity_id"] for s in speakers]
                else:
                    entity_ids = []
                    invalid_entities = []

                    for entity_name in entity_names:
                        if "." in entity_name:
                            try:
                                await ha_client.get_entity(entity_name)
                                entity_ids.append(entity_name)
                            except Exception as e:
                                logger.warning(f"Entity {entity_name} not found: {e}")
                                invalid_entities.append(entity_name)
                        else:
                            invalid_entities.append(entity_name)

                    # If all entities are invalid, try using context-aware last speakers
                    if invalid_entities and not entity_ids:
                        context_speakers = await _get_last_speakers()
                        if context_speakers:
                            logger.info(f"Using context-aware speakers for stop command: {context_speakers}")
                            entity_ids = context_speakers
                        else:
                            return f"Error: The following speaker entities were not found: {', '.join(invalid_entities)}. You must first call list_available_entities with domain='media_player' to get the exact entity_id from Home Assistant. Do not guess or construct entity_ids."

                if not entity_ids:
                    # Final fallback: try context-aware speakers
                    entity_ids = await _get_last_speakers()
                    if not entity_ids:
                        return "Error: No valid speaker entities found. You must first call list_available_entities with domain='media_player' to get available speakers."

                # Call media_stop or media_pause based on action
                service_name = "media_stop" if action == "stop" else "media_pause"
                # #region agent log
                import json
                try:
                    with open('/opt/llm/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A,B,C","location":"ha_executor.py:1034","message":"Before service call","data":{"service_name":service_name,"entity_count":len(entity_ids),"entity_ids":entity_ids},"timestamp":int(__import__('time').time()*1000)}) + "\n")
                except: pass
                # #endregion
                logger.info(f"Calling media_player.{service_name} on {entity_ids}")
                result = await ha_client.call_service(
                    "media_player",
                    service_name,
                    entity_id=entity_ids,
                )
                logger.info(f"media_player.{service_name} service result: {result}")
                # Track speakers for context-aware future commands
                await _track_speakers(entity_ids)
                action_word = "Stopped" if action == "stop" else "Paused"
                return f"{action_word} media on speakers: {', '.join(entity_ids)}"

            elif function_name == "control_media_playback":
                entity_names = args.get("entity_names", [])
                control_action = args.get("control_action", "next_track")
                
                # Get entity IDs - use context-aware last speakers if not provided
                if not entity_names or (len(entity_names) == 1 and entity_names[0].lower() in ["", "none", "null"]):
                    entity_ids = await _get_last_speakers()
                    if not entity_ids:
                        return "Error: No speakers specified and no previous speakers found. Please specify which speakers to control."
                else:
                    # Check if "all" is requested
                    is_all_requested = (
                        "all" in [name.lower() for name in entity_names] or
                        any("all" in name.lower() for name in entity_names)
                    )
                    
                    if is_all_requested:
                        speakers = await ha_client.find_speakers()
                        speakers = _filter_dummy_entities(speakers)
                        entity_ids = [s["entity_id"] for s in speakers]
                    else:
                        entity_ids = []
                        invalid_entities = []
                        
                        for entity_name in entity_names:
                            if "." in entity_name:
                                try:
                                    await ha_client.get_entity(entity_name)
                                    entity_ids.append(entity_name)
                                except Exception as e:
                                    logger.warning(f"Entity {entity_name} not found: {e}")
                                    invalid_entities.append(entity_name)
                            else:
                                invalid_entities.append(entity_name)
                        
                        # If all entities are invalid, try using context-aware last speakers
                        if invalid_entities and not entity_ids:
                            context_speakers = await _get_last_speakers()
                            if context_speakers:
                                logger.info(f"Using context-aware speakers for playback control: {context_speakers}")
                                entity_ids = context_speakers
                            else:
                                return f"Error: The following speaker entities were not found: {', '.join(invalid_entities)}. You must first call list_available_entities with domain='media_player' to get the exact entity_id from Home Assistant."
                
                if not entity_ids:
                    return "Error: No valid speaker entities found."
                
                # Map control_action to Home Assistant service
                service_map = {
                    "next_track": "media_next_track",
                    "previous_track": "media_previous_track",
                    "skip": "media_next_track",
                    "back": "media_previous_track",
                }
                
                service_name = service_map.get(control_action, "media_next_track")
                logger.info(f"Calling media_player.{service_name} on {entity_ids}")
                
                try:
                    result = await ha_client.call_service(
                        "media_player",
                        service_name,
                        entity_id=entity_ids,
                    )
                    logger.info(f"media_player.{service_name} service result: {result}")
                    # Track speakers for context-aware future commands
                    await _track_speakers(entity_ids)
                    
                    action_words = {
                        "next_track": "Skipped to next track",
                        "previous_track": "Skipped to previous track",
                        "skip": "Skipped to next track",
                        "back": "Skipped to previous track",
                    }
                    action_word = action_words.get(control_action, "Changed track")
                    return f"{action_word} on speakers: {', '.join(entity_ids)}"
                except Exception as e:
                    logger.error(f"Error controlling playback: {e}", exc_info=True)
                    return f"Error controlling playback: {str(e)}"

            elif function_name == "set_volume_on_speakers":
                entity_names = args.get("entity_names", [])
                volume_level = args.get("volume_level")
                logger.info(f"[Volume] set_volume_on_speakers called with entity_names={entity_names}, volume_level={volume_level}")

                if volume_level is None:
                    return "Error: volume_level is required"

                # Parse volume_level - handle both numeric and percentage strings
                if isinstance(volume_level, str):
                    volume_str = volume_level.replace("%", "").strip()
                    try:
                        volume_num = float(volume_str)
                        # If the number is > 1, assume it's a percentage (e.g., "60" = 60%)
                        # Otherwise assume it's already a decimal (e.g., "0.6" = 0.6)
                        if volume_num > 1.0:
                            volume_float = volume_num / 100.0
                        else:
                            volume_float = volume_num
                    except ValueError:
                        return f"Error: Invalid volume level format: {volume_level}. Use a percentage like '60%' or a decimal like '0.6'."
                else:
                    # Handle numeric type (backward compatibility)
                    volume_float = float(volume_level)

                # Clamp to 0-1 range
                volume_float = max(0.0, min(1.0, volume_float))

                # Get entity IDs - use last interacted speakers if not provided
                if not entity_names or (len(entity_names) == 1 and entity_names[0].lower() in ["", "none", "null"]):
                    # Use context-aware last speakers
                    entity_ids = await _get_last_speakers()
                    if not entity_ids:
                        return "Error: No speakers specified and no previous speaker interaction found. Please specify which speakers to set volume on, or interact with speakers first (play/stop music)."
                    logger.info(f"[Volume] Using context-aware last speakers: {entity_ids}")
                elif "all" in [name.lower() for name in entity_names]:
                    speakers = await ha_client.find_speakers()
                    speakers = _filter_dummy_entities(speakers)
                    entity_ids = [s["entity_id"] for s in speakers]
                else:
                    entity_ids = []
                    invalid_entities = []

                    for entity_name in entity_names:
                        if "." in entity_name:
                            try:
                                await ha_client.get_entity(entity_name)
                                entity_ids.append(entity_name)
                            except Exception as e:
                                logger.warning(f"Entity {entity_name} not found: {e}")
                                invalid_entities.append(entity_name)
                        else:
                            invalid_entities.append(entity_name)

                    if invalid_entities:
                        return f"Error: The following speaker entities were not found: {', '.join(invalid_entities)}. You must first call list_available_entities with domain='media_player' to get the exact entity_id from Home Assistant. Do not guess or construct entity_ids."

                if not entity_ids:
                    return "Error: No valid speaker entities found. You must first call list_available_entities with domain='media_player' to get available speakers."

                # Call volume_set service
                logger.info(f"Calling media_player.volume_set on {entity_ids} with volume={volume_float}")
                result = await ha_client.call_service(
                    "media_player",
                    "volume_set",
                    entity_id=entity_ids,
                    volume_level=volume_float,
                )
                logger.info(f"media_player.volume_set service result: {result}")
                # Track speakers for context-aware future commands
                await _track_speakers(entity_ids)
                volume_percent = int(volume_float * 100)
                return f"Set volume to {volume_percent}% on speakers: {', '.join(entity_ids)}"

            elif function_name == "add_item_to_shopping_list":
                item_name = args.get("item_name", "")
                if not item_name:
                    return "Error: No item name provided"

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

                    if isinstance(result, list):
                        return f"Added {item_name} to shopping list"
                    elif isinstance(result, dict):
                        if result.get("status") == "error" or "error" in result:
                            logger.error(f"Error response from todo.add_item: {result}")
                            return f"Error: Failed to add {item_name} to shopping list. Response: {result}"
                        else:
                            return f"Added {item_name} to shopping list"
                    else:
                        logger.warning(f"Unexpected result type from todo.add_item: {type(result)}, value: {result}")
                        return f"Added {item_name} to shopping list"

                except Exception as e:
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
                entity_id = "todo.google_keep_shopping"
                logger.info(f"Getting shopping list items from entity '{entity_id}' using todo.get_items service")
                try:
                    service_data = {
                        "entity_id": entity_id,
                        "status": ["needs_action"]
                    }
                    logger.info(f"Calling todo.get_items with data: {json.dumps(service_data)}")

                    result = await ha_client.call_service(
                        "todo",
                        "get_items",
                        return_response=True,
                        **service_data
                    )
                    logger.info(f"todo.get_items service call result type: {type(result)}")
                    logger.debug(f"todo.get_items service call result: {json.dumps(result, indent=2)}")

                    items = []
                    if isinstance(result, dict):
                        service_response = result.get("service_response", {})
                        if entity_id in service_response:
                            entity_data = service_response[entity_id]
                            if isinstance(entity_data, dict) and "items" in entity_data:
                                items = entity_data["items"]
                                logger.info(f"Found {len(items)} items in service response")
                    elif isinstance(result, list):
                        items = result

                    if not items:
                        return "Shopping list is empty."

                    item_names = []
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get("summary") or item.get("item") or item.get("name")
                            status = item.get("status")
                            if status and status != "needs_action":
                                continue
                            if name:
                                item_names.append(str(name))
                        else:
                            item_names.append(str(item))

                    if not item_names:
                        return "Shopping list is empty."

                    return "Shopping list:\n" + "\n".join(f"- {name}" for name in item_names)

                except Exception as e:
                    logger.error(f"Error getting shopping list items: {e}", exc_info=True)
                    return f"Error retrieving shopping list: {str(e)}"

            elif function_name == "start_timer":
                duration_minutes = args.get("duration_minutes")
                name = args.get("name")
                if duration_minutes is None:
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
                entity_id = args.get("entity_id")
                if not entity_id:
                    return "Error: entity_id is required for get_entity_state"
                result = await ha_client.get_entity(entity_id)
                return json.dumps(result)

            elif function_name == "list_available_entities":
                domain = args.get("domain")
                query = args.get("query")
                listing = await _get_or_refresh_listing(domain, ha_client, query=query)
                return listing

            elif function_name == "turn_on_entity":
                entity_id = args.get("entity_id")
                if not entity_id:
                    return "Error: entity_id is required for turn_on_entity"
                try:
                    await ha_client.get_entity(entity_id)
                except Exception:
                    return (
                        f"Error: entity_id '{entity_id}' not found. "
                        "List entities (light and switch) first, then use the exact entity_id."
                    )
                await ha_client.call_service("homeassistant", "turn_on", entity_id=entity_id)
                return f"Turned on {entity_id}"

            elif function_name == "turn_off_entity":
                entity_id = args.get("entity_id")
                if not entity_id:
                    return "Error: entity_id is required for turn_off_entity"
                try:
                    await ha_client.get_entity(entity_id)
                except Exception:
                    return (
                        f"Error: entity_id '{entity_id}' not found. "
                        "List entities (light and switch) first, then use the exact entity_id."
                    )
                await ha_client.call_service("homeassistant", "turn_off", entity_id=entity_id)
                return f"Turned off {entity_id}"

            elif function_name == "find_phone_blasxel_6":
                result = await ha_client.call_service(
                    "script",
                    "turn_on",
                    entity_id="script.find_phone_blasxel_6"
                )
                return f"Triggered phone finder: {result}"

            elif function_name == "refresh_entity_cache":
                await refresh_entity_cache(ha_client)
                return "Entity cache refreshed."

            else:
                return f"Unknown function: {function_name}"

    except Exception as e:
        logger.error(f"Error executing HA function {function_name}: {e}", exc_info=True)
        return f"Error executing {function_name}: {str(e)}"


async def _get_or_refresh_listing(domain: Optional[str], ha_client, query: Optional[str] = None) -> str:
    """Get cached listing, refresh from HA if missing or query provided."""
    # If query is provided, we need to filter; prefer using cached entities if available
    if query:
        listing = await _get_cached_listing(domain)
        if listing:
            return _filter_listing(listing, query)
    # No query or cache miss: refresh
    await refresh_entity_cache(ha_client, domains=[domain] if domain else None)
    listing = await _get_cached_listing(domain)
    if query and listing:
        return _filter_listing(listing, query)
    if listing:
        return listing
    domain_note = f"in domain '{domain}' " if domain else ""
    return f"Found 0 entities {domain_note}in Home Assistant."


def _filter_listing(listing: str, query: str) -> str:
    filtered = []
    for line in listing.splitlines():
        if query.lower() in line.lower():
            filtered.append(line)
    if not filtered:
        return f"Found 0 entities matching '{query}'."
    header = f"Found {len(filtered)} entities matching '{query}':"
    return header + "\n" + "\n".join(filtered)


async def _get_cached_listing(domain: Optional[str]) -> Optional[str]:
    async with _CACHE_LOCK:
        if domain:
            return _ENTITY_CACHE.get(domain)
        # If no domain, combine all
        if _ENTITY_CACHE:
            parts = []
            for d, listing in _ENTITY_CACHE.items():
                parts.append(f"{d} domain:\n{listing}")
            return "\n\n".join(parts)
        return None


async def refresh_entity_cache(ha_client, domains: Optional[List[str]] = None) -> None:
    """Refresh entity listings for specified domains (or default set) and store in cache."""
    target_domains = domains or ["light", "switch", "lock", "media_player"]
    for domain in target_domains:
        try:
            # Ensure the HA client session is opened and closed for each refresh
            async with ha_client:
                async with _CACHE_LOCK:
                    pass  # serialize per-domain refresh
                entities = await ha_client.get_entities()
                entities = _filter_dummy_entities(entities)
                if domain:
                    entities = [e for e in entities if e.get("entity_id", "").startswith(f"{domain}.")]
                results = []
                for entity in entities:
                    entity_id = entity.get("entity_id", "")
                    friendly_name = entity.get("attributes", {}).get("friendly_name", "")
                    results.append(f"{friendly_name} ({entity_id})" if friendly_name else entity_id)
                header = f"Found {len(results)} entities"
                if domain:
                    header += f" in domain '{domain}'"
                listing = header + ":\n" + "\n".join(results) if results else f"Found 0 entities in domain '{domain}'."
                async with _CACHE_LOCK:
                    _ENTITY_CACHE[domain] = listing
        except Exception as e:
            logger.warning(f"Failed to refresh entity cache for domain {domain}: {e}")


async def _prefetch_light_switch_listings(ha_client) -> List[Dict[str, Any]]:
    """Get cached or refreshed light/switch listings and return as tool messages to steer the first turn."""
    listings: List[Dict[str, Any]] = []
    for domain in ["light", "switch"]:
        listing = await _get_or_refresh_listing(domain, ha_client)
        if listing:
            listings.append(
                {
                    "role": "tool",
                    "name": "list_available_entities",
                    "content": listing,
                    "tool_call_id": f"prefetch_{domain}",
                }
            )
    return listings


async def _prefetch_media_player_listings(ha_client) -> List[Dict[str, Any]]:
    """Get cached or refreshed media_player listings and return as tool messages to steer the first turn."""
    listings: List[Dict[str, Any]] = []
    listing = await _get_or_refresh_listing("media_player", ha_client)
    if listing:
        listings.append(
            {
                "role": "tool",
                "name": "list_available_entities",
                "content": listing,
                "tool_call_id": "prefetch_media_player",
            }
        )
    return listings


async def start_entity_cache_refresher(ha_client):
    """Start a daily background task to refresh the entity cache."""
    global _REFRESH_TASK
    if _REFRESH_TASK and not _REFRESH_TASK.done():
        return

    async def _runner():
        while True:
            try:
                await refresh_entity_cache(ha_client)
            except Exception as e:
                logger.warning(f"Background cache refresh failed: {e}")
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)

    _REFRESH_TASK = asyncio.create_task(_runner())


async def stop_entity_cache_refresher():
    """Stop the background cache refresh task."""
    global _REFRESH_TASK
    if _REFRESH_TASK:
        _REFRESH_TASK.cancel()
        try:
            await _REFRESH_TASK
        except asyncio.CancelledError:
            pass
        _REFRESH_TASK = None

