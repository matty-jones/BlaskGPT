"""
Home Assistant Function Definitions for OpenAI Function Calling

Defines function schemas that the LLM can call to interact with Home Assistant.
"""

from typing import List, Optional, Dict, Any


# Function schemas in OpenAI function calling format
HA_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "play_media_on_speakers",
            "description": "Play media (music, podcast, etc.) on one or more speakers. Use 'all' or 'all speakers' to target all available speakers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of speaker entity names or IDs. Use 'all' to target all speakers."
                    },
                    "media_content_id": {
                        "type": "string",
                        "description": "The media to play. For music, this could be a song name, artist, or playlist. For podcasts, the episode name."
                    },
                    "media_content_type": {
                        "type": "string",
                        "enum": ["music", "podcast", "playlist", "album"],
                        "description": "Type of media content",
                        "default": "music"
                    }
                },
                "required": ["entity_names", "media_content_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_item_to_shopping_list",
            "description": "Add an item to the Home Assistant shopping list. Use this when the user explicitly wants to ADD an item to the list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The name of the item to add to the shopping list"
                    }
                },
                "required": ["item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_shopping_list_items",
            "description": "Get the current items on the shopping list. Use this when the user asks about the shopping list (e.g., 'what's on my shopping list?', 'how many items?', 'is X on the list?', 'show me the shopping list', 'read me the items'). IMPORTANT: When the user asks 'what is on' or 'read me' or 'show me' the shopping list, you MUST return the FULL LIST of items, not just the count. The function returns all items with their names - present them all to the user. This function takes no parameters.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_timer",
            "description": "Start a timer for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the timer in minutes"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the timer"
                    }
                },
                "required": ["duration_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_state",
            "description": "Get the current state of a Home Assistant entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID (e.g., 'media_player.living_room_speaker')"
                    }
                },
                "required": ["entity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_entities",
            "description": "List all available Home Assistant entities, optionally filtered by domain. When the user wants to control an entity by description (e.g., 'lock the car'), call this with ONLY the domain parameter (e.g., domain='lock') - DO NOT use the query parameter. Get ALL entities in that domain, then use semantic understanding to match the description. The query parameter only does literal string matching, which won't work for semantic matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Optional domain filter (e.g., 'media_player', 'light', 'switch')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query to filter entities by name"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_on_entity",
            "description": "Turn on a Home Assistant entity (light, switch, etc.) or LOCK a lock entity. IMPORTANT: For lock entities, 'lock' means to secure/lock them - use turn_on_entity. Examples: 'lock the car' → turn_on_entity, 'lock the door' → turn_on_entity. For lights/switches, 'turn on' → turn_on_entity. CRITICAL: You MUST first call list_available_entities to find the exact entity_id if the user provides a natural language name (e.g., 'den lights'). Do NOT guess or construct entity_ids like 'light.den' - you must search for the actual entity_id first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The exact entity ID from Home Assistant (e.g., 'switch.den_lights', 'light.living_room'). Must be obtained from list_available_entities first if user provides a natural language name."
                    }
                },
                "required": ["entity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "turn_off_entity",
            "description": "Turn off a Home Assistant entity (light, switch, etc.) or UNLOCK a lock entity. IMPORTANT: For lock entities, 'unlock' means to open/unlock them - use turn_off_entity. Examples: 'unlock the car' → turn_off_entity, 'unlock the door' → turn_off_entity. For lights/switches, 'turn off' → turn_off_entity. CRITICAL: You MUST first call list_available_entities to find the exact entity_id if the user provides a natural language name (e.g., 'den lights'). Do NOT guess or construct entity_ids like 'light.den' - you must search for the actual entity_id first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The exact entity ID from Home Assistant (e.g., 'switch.den_lights', 'light.living_room'). Must be obtained from list_available_entities first if user provides a natural language name."
                    }
                },
                "required": ["entity_id"]
            }
        }
    }
]


# Function name to handler mapping
FUNCTION_HANDLERS = {
    "play_media_on_speakers": "handle_play_media",
    "add_item_to_shopping_list": "handle_add_shopping_item",
    "get_shopping_list_items": "handle_get_shopping_list_items",
    "start_timer": "handle_start_timer",
    "get_entity_state": "handle_get_entity_state",
    "list_available_entities": "handle_list_entities",
    "turn_on_entity": "handle_turn_on",
    "turn_off_entity": "handle_turn_off"
}


def get_function_schemas() -> List[Dict[str, Any]]:
    """Get all function schemas"""
    return HA_FUNCTIONS


def get_function_names() -> List[str]:
    """Get list of all function names"""
    return [func["function"]["name"] for func in HA_FUNCTIONS]
