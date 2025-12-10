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
            "name": "search_music",
            "description": "Search and auto-play music. Selects best match via fuzzy matching. Provide raw_utterance for type/artist hints. If no strong match, asks for clarification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Music search query (artist, song, album, etc.)"
                    },
                    "raw_utterance": {
                        "type": "string",
                        "description": "Original user message for extracting type/artist hints"
                    },
                    "area": {
                        "type": "string",
                        "description": "Optional speaker name. If omitted, uses last interacted speakers.",
                        "default": ""
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["artist", "track", "album", "playlist", "any"],
                        "description": "Optional type hint. System also extracts from raw_utterance.",
                        "default": "any"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max search results. Default 10. Use 20-50 for random songs.",
                        "default": 10
                    }
                },
                "required": ["query", "raw_utterance"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_media_on_speakers",
            "description": "Play media on speakers. Use list_available_entities to find exact entity_id from natural language names. For music, use search_music first to get library:// URIs. Never guess entity_ids or URIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Speaker entity IDs. Use list_available_entities for natural names. Use 'all' for all speakers."
                    },
                    "media_content_id": {
                        "type": "string",
                        "description": "Media identifier (e.g., 'library://track/456'). Get from search_music. Never guess URIs."
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
            "name": "stop_media_on_speakers",
            "description": "Stop or pause media playback. Use list_available_entities to find exact entity_id from natural language names. Use 'all' for all speakers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Speaker entity IDs. Use list_available_entities for natural names. Use 'all' for all speakers."
                    },
                    "action": {
                        "type": "string",
                        "enum": ["stop", "pause"],
                        "description": "Whether to stop (completely stop) or pause (can resume) the media",
                        "default": "stop"
                    }
                },
                "required": ["entity_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume_on_speakers",
            "description": "Set volume level. If no speakers specified, uses last interacted speakers. Use list_available_entities for natural language names. Volume: '60%' or '0.6'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Speaker entity IDs. If empty, uses last interacted speakers. Use 'all' for all speakers."
                    },
                    "volume_level": {
                        "type": "string",
                        "description": "Volume level as a percentage string (e.g., '60%', '30%', '100%') or a decimal number as a string (e.g., '0.6' for 60%). The system will parse and convert percentages to the 0-1 range automatically."
                    }
                },
                "required": ["volume_level"]
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
            "description": "Get shopping list items. Returns full list - show all items to user, not just count.",
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
            "description": "List entities, optionally by domain. For semantic matching (e.g., 'lock the car'), use domain only, not query. Get all entities in domain, then match semantically.",
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
            "description": "Turn on entity or LOCK lock. Use list_available_entities first for natural language names. Never guess entity_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID (e.g., 'switch.den_lights'). Get from list_available_entities for natural names."
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
            "description": "Turn off entity or UNLOCK lock. Use list_available_entities first for natural language names. Never guess entity_ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Entity ID (e.g., 'switch.den_lights'). Get from list_available_entities for natural names."
                    }
                },
                "required": ["entity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_phone_blasxel_6",
            "description": "Ring Blasxel 6 phone until unlocked, then restore volume.",
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
            "name": "refresh_entity_cache",
            "description": "Refresh cached entity listings for faster future lookups.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# Function name to handler mapping
FUNCTION_HANDLERS = {
    "search_music": "handle_search_music",
    "play_media_on_speakers": "handle_play_media",
    "add_item_to_shopping_list": "handle_add_shopping_item",
    "get_shopping_list_items": "handle_get_shopping_list_items",
    "start_timer": "handle_start_timer",
    "get_entity_state": "handle_get_entity_state",
    "list_available_entities": "handle_list_entities",
    "turn_on_entity": "handle_turn_on",
    "turn_off_entity": "handle_turn_off",
    "find_phone_blasxel_6": "handle_find_phone_blasxel_6",
    "refresh_entity_cache": "handle_refresh_entity_cache",
}


def get_function_schemas() -> List[Dict[str, Any]]:
    """Get all function schemas"""
    return HA_FUNCTIONS


def get_function_schemas_by_names(names: List[str]) -> List[Dict[str, Any]]:
    """Get function schemas filtered by function name"""
    name_set = set(names)
    return [func for func in HA_FUNCTIONS if func["function"]["name"] in name_set]


def get_function_names() -> List[str]:
    """Get list of all function names"""
    return [func["function"]["name"] for func in HA_FUNCTIONS]
