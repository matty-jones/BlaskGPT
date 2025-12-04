#!/usr/bin/env python3
"""
Simple CLI tool to interact with the LLM chat API

Usage:
    python chat.py "What media players do I have?"
    python chat.py --use-case ha_command "Add milk to shopping list"
    python chat.py --use-case googling "What is the capital of France?"
"""

import argparse
import json
import sys
from typing import Optional
import httpx


def send_chat_request(
    message: str,
    use_case: Optional[str] = None,
    base_url: str = "http://localhost:8080"
) -> dict:
    """Send a chat request to the API gateway using OpenAI-compatible format"""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": message}]
    }
    if use_case:
        payload["use_case"] = use_case
    
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Error: Failed to connect to API gateway at {base_url}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def format_response(response: dict) -> str:
    """Format the response nicely for terminal output"""
    output = []
    
    # Extract response from OpenAI format
    if "choices" in response and len(response["choices"]) > 0:
        message = response["choices"][0].get("message", {})
        response_text = message.get("content", "")
        if response_text:
            output.append("Response:")
            output.append("=" * 70)
            output.append(response_text)
            output.append("")
    else:
        # Fallback for old format
        response_text = response.get("response", "")
        if response_text:
            output.append("Response:")
            output.append("=" * 70)
            output.append(response_text)
            output.append("")
    
    # Metadata
    model = response.get("model", "unknown")
    usage = response.get("usage", {})
    
    output.append("Metadata:")
    output.append("-" * 70)
    output.append(f"  Model: {model}")
    if usage:
        output.append(f"  Tokens: {usage.get('total_tokens', 'N/A')} (prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')})")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Test the LLM chat API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What media players do I have?"
  %(prog)s --use-case ha_command "Add milk to shopping list"
  %(prog)s --use-case googling "What is the speed of light?"
  %(prog)s --use-case automation_variation "Lock the car"
  %(prog)s --url http://192.168.1.100:8080 "Hello"
        """
    )
    
    parser.add_argument(
        "message",
        help="The message to send to the LLM"
    )
    
    parser.add_argument(
        "--use-case",
        choices=["ha_command", "googling", "automation_variation", "general"],
        help="Specify the use case (otherwise auto-detected)"
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the API gateway (default: http://localhost:8080). The endpoint /v1/chat/completions will be used."
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )
    
    args = parser.parse_args()
    
    # Send request
    response = send_chat_request(
        message=args.message,
        use_case=args.use_case,
        base_url=args.url
    )
    
    # Output
    if args.json:
        print(json.dumps(response, indent=2))
    else:
        print(format_response(response))


if __name__ == "__main__":
    main()

