"""
HA sub-use-case routing, prompts, and tool selection.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from tools.ha_functions import get_function_names


class HASubUseCase(str, Enum):
    LIGHTS_SWITCHES_LOCKS = "lights_switches_locks"
    MUSIC = "music"
    SHOPPING = "shopping_list"
    TIMERS = "timers"
    STATUS = "status"
    PHONE = "phone_find"
    MISC = "other_script"


PROMPT_FILES: Dict[HASubUseCase, str] = {
    HASubUseCase.LIGHTS_SWITCHES_LOCKS: "ha_lights_switches_locks.txt",
    HASubUseCase.MUSIC: "ha_music.txt",
    HASubUseCase.SHOPPING: "ha_shopping.txt",
    HASubUseCase.TIMERS: "ha_timers.txt",
    HASubUseCase.STATUS: "ha_status.txt",
    HASubUseCase.PHONE: "ha_phone.txt",
    HASubUseCase.MISC: "ha_misc.txt",
}

# Tool subsets by sub-use-case
TOOLS_BY_SUBCASE: Dict[HASubUseCase, List[str]] = {
    HASubUseCase.LIGHTS_SWITCHES_LOCKS: [
        "list_available_entities",
        "turn_on_entity",
        "turn_off_entity",
    ],
    HASubUseCase.MUSIC: [
        "list_available_entities",
        "search_music",
        "play_media_on_speakers",
        "stop_media_on_speakers",
        "set_volume_on_speakers",
    ],
    HASubUseCase.SHOPPING: [
        "add_item_to_shopping_list",
        "get_shopping_list_items",
    ],
    HASubUseCase.TIMERS: [
        "start_timer",
    ],
    HASubUseCase.STATUS: [
        "list_available_entities",
        "get_entity_state",
    ],
    HASubUseCase.PHONE: [
        "find_phone_blasxel_6",
    ],
    HASubUseCase.MISC: get_function_names(),  # fallback: allow all
}

# Prompt search locations
PROMPT_BASES = [
    Path("/app/prompts/ha"),
    Path("/opt/llm/prompts/ha"),
]
CORE_PROMPT_PATHS = [
    Path("/app/prompts/ha/core_preamble.txt"),
    Path("/opt/llm/prompts/ha/core_preamble.txt"),
]


def classify_ha_sub_use_case(message: str) -> HASubUseCase:
    """Lightweight rules to route HA commands to sub-use-cases."""
    text = message.lower()

    # Music intent
    music_keys = ["play", "shuffle", "song", "album", "playlist", "music", "tune", "podcast", "speaker", "speakers"]
    if any(k in text for k in music_keys):
        return HASubUseCase.MUSIC

    # Shopping list
    shopping_keys = ["shopping list", "add to list", "grocer", "grocery", "buy", "shopping"]
    if any(k in text for k in shopping_keys):
        return HASubUseCase.SHOPPING

    # Timers
    if "timer" in text or "countdown" in text:
        return HASubUseCase.TIMERS

    # Phone find
    if "find my phone" in text or "ring my phone" in text or "find the phone" in text or "blasxel" in text:
        return HASubUseCase.PHONE

    # Status / listing queries
    status_patterns = ["what", "list", "show", "status", "state", "is the", "are the"]
    question_words = ["what", "which", "show", "list", "is", "are", "status", "state"]
    if any(text.startswith(word) for word in question_words) or any(key in text for key in status_patterns):
        return HASubUseCase.STATUS

    # Locks / lights / switches default for action phrasing
    control_terms = ["turn on", "turn off", "lock", "unlock", "lights", "switch", "switches", "light"]
    if any(k in text for k in control_terms):
        return HASubUseCase.LIGHTS_SWITCHES_LOCKS

    # Fallback
    return HASubUseCase.MISC


def get_tools_for_sub_use_case(sub_use_case: HASubUseCase) -> List[str]:
    return TOOLS_BY_SUBCASE.get(sub_use_case, TOOLS_BY_SUBCASE[HASubUseCase.MISC])


def _load_from_paths(paths: List[Path]) -> Optional[str]:
    for path in paths:
        try:
            return path.read_text()
        except FileNotFoundError:
            continue
    return None


def build_system_prompt(sub_use_case: HASubUseCase) -> str:
    """Compose the core preamble with the sub-use-case prompt."""
    core = _load_from_paths(CORE_PROMPT_PATHS) or ""
    sub_prompt_name = PROMPT_FILES.get(sub_use_case)
    sub_prompt = ""
    if sub_prompt_name:
        sub_paths = [base / sub_prompt_name for base in PROMPT_BASES]
        sub_prompt = _load_from_paths(sub_paths) or ""
    prompt_parts = [p.strip() for p in [core, sub_prompt] if p]
    return "\n\n".join(prompt_parts)

