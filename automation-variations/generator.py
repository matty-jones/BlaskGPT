"""
Automation Variation Generator

Generates creative variations of Home Assistant automation messages using the LLM.
"""

import logging
from typing import List, Optional
import httpx
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Settings for variation generator"""
    vllm_url: str = "http://vllm:8000"
    vllm_api_key: str = "local-dev-key"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


def load_prompt_template() -> str:
    """Load the automation variation prompt template"""
    try:
        with open("/opt/llm/prompts/automation_variation.txt", "r") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load prompt template: {e}")
        return """You are a creative assistant that generates variations of automation messages.
Generate creative variations while maintaining the same intent and meaning."""


async def generate_variations(
    base_message: str,
    style: str = "humorous",
    count: int = 1,
    settings: Optional[Settings] = None
) -> List[str]:
    """
    Generate variations of an automation message
    
    Args:
        base_message: The original message to vary
        style: Style of variation (humorous, formal, casual, friendly)
        count: Number of variations to generate
        settings: Optional settings instance
    
    Returns:
        List of message variations
    """
    if settings is None:
        settings = Settings()
    
    system_prompt = load_prompt_template()
    
    user_prompt = f"""Generate {count} variation(s) of this message in a {style} style:

Base message: "{base_message}"

Provide {count} variation(s), one per line, without numbering or bullets."""

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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.8,  # Higher temperature for more creativity
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse variations from response
            variations = _parse_variations(content, count)
            
            logger.info(f"Generated {len(variations)} variations for: {base_message[:50]}")
            return variations
    
    except Exception as e:
        logger.error(f"Error generating variations: {e}", exc_info=True)
        # Fallback: return the base message
        return [base_message] * count


def _parse_variations(content: str, expected_count: int) -> List[str]:
    """
    Parse variations from LLM response
    
    Args:
        content: LLM response text
        expected_count: Expected number of variations
    
    Returns:
        List of parsed variations
    """
    # Split by newlines and clean up
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Remove common prefixes like "1.", "-", "*", etc.
    cleaned = []
    for line in lines:
        # Remove numbered prefixes
        if line and line[0].isdigit() and len(line) > 2 and line[1] in ['.', ')', ':']:
            line = line[2:].strip()
        # Remove bullet points
        if line.startswith('- ') or line.startswith('* '):
            line = line[2:].strip()
        # Remove quotes if present
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1]
        
        if line:
            cleaned.append(line)
    
    # Return up to expected_count variations
    return cleaned[:expected_count] if cleaned else [content]
