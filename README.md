# Local LLM Integration for Home Assistant

This project provides a locally-hosted large language model (LLM) integration for Home Assistant, running on your workstation's GPU. It uses OpenAI-compatible API endpoints and supports intelligent routing to specialized handlers for different use cases.

## Features

1. **Interactive Googling**: Answer factual questions using web search with concise responses
2. **Home Assistant Commands**: Control your smart home through natural language with intelligent sub-use-case routing:
   - **Lights, Switches, and Locks**: Turn devices on/off, lock/unlock with semantic entity matching
   - **Music Control**: Search and play music via Music Assistant with fuzzy matching, context-aware speaker tracking, volume control, and playback controls
   - **Shopping List**: Add items and retrieve shopping list contents
   - **Timers**: Start timers with specified durations
   - **Entity Status**: Query device states and list available entities
   - **Phone Finder**: Trigger phone finding scripts
3. **Automation Variations**: Generate creative variations of automation messages
4. **General Conversation**: Natural language chat with contextual awareness

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  Home Assistant │         │  Web Browser    │
│  (Voice/Webhook)│         │  (BlaskGPT UI)  │
└────────┬────────┘         └────────┬────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────┐ ┌──────────────────────────────┐
│     API Gateway (FastAPI)       │ │  BlaskGPT (OpenWebUI)        │
│  Port 8080                      │ │  Port 3000                   │
│  - Request routing              │ │  - ChatGPT-like UI            │
│  - Use case detection           │ │  - No authentication          │
│  - Context management          │ │  - General-purpose chat       │
└────────┬────────────────────────┘ └────────┬───────────────────────┘
         │                                    │
         └────────────┬───────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  vLLM (Internal)│
              │  Port 8000      │
              │  - Model serving │
              │  - Internal only │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌──────────┐ ┌──────────┐ ┌─────────────┐
    │   Web    │ │    HA    │ │ Automation  │
    │  Search  │ │  Client  │ │  Variations │
    │ Service  │ │          │ │   Engine    │
    └──────────┘ └──────────┘ └─────────────┘
```

**Components**:
- **vLLM**: Serves the Qwen2.5-7B-Instruct-AWQ model via OpenAI-compatible API (internal only, port 8000)
- **API Gateway**: FastAPI service that routes requests and coordinates components (port 8080)
- **BlaskGPT (OpenWebUI)**: General-purpose ChatGPT-like web UI (port 3000, no authentication)
- **HA Client**: Python client for Home Assistant API integration
- **Search Service**: Web search integration using DuckDuckGo
- **Automation Variations**: LLM-powered message variation generator

## Setup

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers and nvidia-docker runtime
- Home Assistant instance with Long-Lived Access Token

### Configuration

1. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   - `HA_URL`: Your Home Assistant URL
   - `HA_ACCESS_TOKEN`: Your Long-Lived Access Token
   - Other settings as needed

### Running

Start all services:
```bash
docker compose up -d
```

Check service status:
```bash
docker compose ps
```

Check logs:
```bash
docker compose logs -f
```

### Health Checks

Verify services are running:
```bash
# Test API Gateway
curl http://localhost:8080/health

# Test BlaskGPT (OpenWebUI)
curl http://localhost:3000
```

**Note**: vLLM is internal-only and not accessible from the LAN. Both API Gateway and BlaskGPT connect to vLLM via the internal Docker network on port 8000.

## API Endpoints

### Chat Completion Endpoint (OpenAI-Compatible)

The API Gateway provides an OpenAI-compatible chat completion endpoint:

```bash
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer local-dev-key (optional, for compatibility)

{
  "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "messages": [
    {"role": "user", "content": "Shuffle Rush on all the speakers"}
  ],
  "use_case": "ha_command"  # Optional: auto-detected if not specified
}
```

**Use Cases** (auto-detected if not specified):
- `googling`: Factual questions that require web search
- `ha_command`: Home Assistant control commands (internally routed to sub-use-cases: lights/switches/locks, music, shopping_list, timers, status, phone_find, other_script)
- `automation_variation`: Generate variations of automation messages
- `general`: General conversation

**Example Requests**:
```bash
# Web search
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "What is the capital of Assyria?"}],
    "use_case": "googling"
  }'

# Home Assistant command
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Add gin to the shopping list"}],
    "use_case": "ha_command"
  }'

# General chat (use_case auto-detected)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello, can you help me?"}]
  }'
```

**Response Format** (OpenAI-compatible):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Response text here"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Variation Generation
```bash
POST /variations
Content-Type: application/json

{
  "base_message": "Bubbers, lock the bloody car",
  "count": 3
}
```

**Note**: The `style` parameter is no longer used. Variations are generated based on the base message context.

### Webhook for Home Assistant
```bash
POST /webhook/variation
Content-Type: application/json

{
  "message": "Your automation message",
  "count": 1
}
```

## BlaskGPT (OpenWebUI) Web Interface

BlaskGPT provides a general-purpose ChatGPT-like web interface for interacting with the local LLM. It runs in parallel with the API Gateway and shares the same vLLM backend.

### Access

- **URL**: `http://<host>:3000`
- **Authentication**: Disabled (single-user mode, LAN access only)
- **Model**: Automatically connects to Qwen2.5-7B-Instruct-AWQ via internal vLLM

### Features

- Natural language chat interface
- Conversation history
- Model selection (shows available models from vLLM)
- No login required (suitable for trusted LAN environments)

### Security Note

BlaskGPT is configured without authentication for ease of use on a trusted LAN. Anyone on your local network can access it. For additional security, consider:
- Placing it behind a reverse proxy with authentication
- Enabling `WEBUI_AUTH=True` in `compose.yml` (requires user registration)
- Using firewall rules to restrict access

## Integration with Home Assistant

### Using the API Gateway

The API Gateway runs on port 8080 and can be accessed from Home Assistant via:

1. **REST API**: Use the `/v1/chat/completions` endpoint in Home Assistant automations or scripts
2. **Webhook**: Use `/webhook/variation` for automation message variations

### Example Home Assistant Automation

```yaml
automation:
  - alias: "Generate varied reminder"
    trigger:
      - platform: state
        entity_id: binary_sensor.car_locked
        to: 'off'
        for: '00:05:00'
    action:
      - service: http.post
        data:
          url: "http://192.168.86.X:8080/webhook/variation"
          method: POST
          headers:
            Content-Type: application/json
          data:
            message: "Bubbers, lock the bloody car"
            count: 1
      - service: notify.mobile_app
        data:
          message: "{{ states('sensor.last_variation') }}"
```

### CLI Tool

A command-line tool is provided for easy testing:

```bash
# Basic usage
python chat.py "What media players do I have?"

# Specify use case
python chat.py --use-case ha_command "Add milk to shopping list"
python chat.py --use-case googling "What is the speed of light?"

# Use different API URL
python chat.py --url http://192.168.1.100:8080 "Hello"

# Output raw JSON
python chat.py --json "Hello"
```

## Development

### Local Development (without Docker)

1. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r api-gateway/requirements.txt
   pip install -r ha-client/requirements.txt
   pip install -r search-service/requirements.txt
   pip install -r automation-variations/requirements.txt
   ```

3. Run API Gateway:
   ```bash
   cd api-gateway
   python main.py
   ```

## Directory Structure

```
/opt/llm/
├── compose.yml                    # Docker Compose configuration
├── .env                           # Environment variables (not in git)
├── chat.py                        # CLI tool for testing the API
├── test_suite.sh                  # Automated test suite
├── api-gateway/                   # Main FastAPI service
│   ├── main.py                    # API Gateway entry point
│   ├── ha_executor.py             # HA command execution with sub-use-case routing
│   ├── ha_router.py               # Sub-use-case classification and tool selection
│   ├── music_matcher.py           # Fuzzy music matching and selection
│   └── requirements.txt
├── ha-client/                     # Home Assistant client library
├── search-service/                # Web search integration (DuckDuckGo)
├── tools/                         # Function calling definitions
│   └── ha_functions.py            # HA function schemas for LLM
├── prompts/                       # System prompts for LLM
│   ├── conversational_system.txt  # General conversation prompt
│   ├── googling_system.txt        # Web search prompt
│   ├── automation_variation.txt   # Variation generation prompt
│   └── ha/                        # Home Assistant sub-use-case prompts
│       ├── core_preamble.txt
│       ├── ha_lights_switches_locks.txt
│       ├── ha_music.txt
│       ├── ha_shopping.txt
│       ├── ha_timers.txt
│       ├── ha_status.txt
│       ├── ha_phone.txt
│       └── ha_misc.txt
├── automation-variations/         # Variation generator
└── home-assistant-scripts/        # Example HA scripts
```

## Key Features and Capabilities

### Home Assistant Integration

- **Intelligent Entity Matching**: Uses semantic matching to resolve natural language device names to entity IDs
- **Entity Caching**: Daily background refresh of entity listings for faster responses
- **Context-Aware Speaker Tracking**: Remembers last interacted speakers for music commands without explicit specification
- **Sub-Use-Case Routing**: Automatically routes commands to specialized handlers:
  - Lights/Switches/Locks: Semantic entity matching, supports both `light` and `switch` domains
  - Music: Fuzzy matching, automatic track selection, artist radio support, volume control, playback controls
  - Shopping List: Google Keep integration via todo entities
  - Timers: Automatic timer selection from available timers
  - Status: Entity state queries and listings
  - Phone Finder: Custom scripts for device location

### Music Control

- **Fuzzy Music Matching**: Intelligent selection of tracks, artists, playlists, and albums using rapidfuzz
- **Music Assistant Integration**: Full support for Music Assistant library URIs
- **Automatic Playback**: Automatically plays selected music on specified or context-aware speakers
- **Track + Artist Radio**: For tracks, automatically plays the track then continues with artist radio
- **Live Version Detection**: Prefers non-live versions by default, but can detect and select live versions when requested
- **Volume Control**: Set volume levels with percentage or decimal format
- **Playback Controls**: Next/previous track, pause/stop with context-aware speaker selection

### System Architecture

- **OpenAI-Compatible API**: Uses standard OpenAI chat completion format for easy integration
- **Tool Calling**: LLM uses function calling to interact with Home Assistant
- **Retry Logic**: Automatic retry with entity listing when entity IDs are not found
- **Entity Prefetching**: Pre-loads entity listings for faster first-turn responses

## Notes

- Voice input/output is handled by Home Assistant directly (Whisper, Piper, etc.)
- All LLM processing happens locally on your GPU
- Web search uses DuckDuckGo (privacy-focused, no API key required)
- The system prefers concise answers for factual queries
- Entity cache refreshes daily in the background
- Music Assistant integration requires Music Assistant to be configured in Home Assistant

## Testing

### Automated Test Suite

Run the comprehensive test suite:
```bash
./test_suite.sh
```

This tests:
- Service health checks
- General chat functionality
- Web search (googling)
- Use case auto-detection
- Automation variations
- Webhook endpoint

### Manual Testing Examples

**Test Home Assistant Commands** (requires valid HA setup):
```bash
# Add item to shopping list
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Add gin to the shopping list"}],
    "use_case": "ha_command"
  }'

# Play media on speakers
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Shuffle Rush on all the speakers"}],
    "use_case": "ha_command"
  }'

# Set timer
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Set a timer for nine minutes"}],
    "use_case": "ha_command"
  }'

# Control lights
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Turn on the den lights"}],
    "use_case": "ha_command"
  }'
```

**Test Web Search**:
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Can dogs eat mushrooms?"}],
    "use_case": "googling"
  }'
```

**Test Automation Variations**:
```bash
curl -X POST http://localhost:8080/variations \
  -H "Content-Type: application/json" \
  -d '{"base_message": "Bubbers, lock the bloody car", "count": 3}'
```

**Using the CLI Tool**:
```bash
# Simple query
python chat.py "What media players do I have?"

# Home Assistant command
python chat.py --use-case ha_command "Turn on the living room lights"

# Web search
python chat.py --use-case googling "What is the speed of light?"
```

## Troubleshooting

### Services Not Starting
- Check Docker: `docker --version`
- Check NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`
- Check logs: `docker compose logs`

### Service Health Issues
- Check Docker logs: `docker compose logs api-gateway`
- Check vLLM logs: `docker compose logs vllm`
- Check BlaskGPT logs: `docker compose logs blaskgpt`
- Verify HA connection: Check API Gateway startup logs
- Test API Gateway: `curl http://localhost:8080/health`
- Test BlaskGPT: `curl http://localhost:3000`

### vLLM Connection Issues
- vLLM is internal-only and not accessible from LAN (by design)
- Check API key matches in `.env` and `compose.yml` (should be `local-dev-key`)
- Verify network connectivity between containers: `docker network inspect llm_llm-network`
- Check that both API Gateway and BlaskGPT can reach vLLM internally: `docker compose exec api-gateway curl http://vllm:8000/health`

### Home Assistant Connection Issues
- Verify HA URL is correct and accessible
- Test HA API directly: `curl -H "Authorization: Bearer YOUR_TOKEN" http://HA_URL/api/config`
- Check access token has required permissions
- Verify network connectivity (same network/subnet)

### GPU Memory Issues
If vLLM fails to start with out of memory errors:
- Check GPU memory: `nvidia-smi`
- Review GPU memory settings in `compose.yml` (currently set to 70% utilization)
- Adjust `--gpu-memory-utilization` in `compose.yml` if needed (lower value = less memory usage)

### Music Assistant Issues
- Ensure Music Assistant is installed and configured in Home Assistant
- Verify Music Assistant entities are available (check for `_2` suffix entities)
- Check that `music_assistant.search` service is available
- Review API Gateway logs for Music Assistant search errors

### Entity Resolution Issues
- The system uses semantic matching to resolve entity names
- If an entity is not found, the system will automatically retry with entity listings
- Check that entity names match friendly names in Home Assistant
- Verify entities are not filtered as "dummy" entities (entities with "dummy" in name are filtered)
