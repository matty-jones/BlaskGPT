# Local LLM Integration for Home Assistant

This project provides a locally-hosted large language model (LLM) integration for Home Assistant, running on your workstation's GPU.

## Features

1. **Interactive Googling**: Answer factual questions using web search with concise responses
2. **Home Assistant Commands**: Control your smart home through natural language
3. **Automation Variations**: Generate creative variations of automation messages

## Architecture

```
┌─────────────────┐
│  Home Assistant │
│  (Voice/Webhook)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     API Gateway (FastAPI)       │
│  Port 8080                      │
│  - Request routing              │
│  - Use case detection           │
│  - Context management           │
└────────┬────────────────────────┘
         │
    ┌────┴────┬──────────────┬─────────────┐
    │         │              │             │
    ▼         ▼              ▼             ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
│  vLLM  │ │   Web    │ │    HA    │ │ Automation  │
│  API   │ │  Search  │ │  Client  │ │  Variations │
│ :9000  │ │ Service  │ │          │ │   Engine    │
└────────┘ └──────────┘ └──────────┘ └─────────────┘
```

**Components**:
- **vLLM**: Serves the Qwen2.5-7B-Instruct-AWQ model via OpenAI-compatible API (external port 9000)
- **API Gateway**: FastAPI service that routes requests and coordinates components (port 8080)
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
# Test vLLM (external access)
curl http://localhost:9000/health

# Test API Gateway
curl http://localhost:8080/health
```

**Note**: vLLM is accessible externally on port 9000, but internally uses port 8000 for Docker network communication.

## API Endpoints

### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
  "message": "Shuffle Rush on all the speakers",
  "use_case": "ha_command"  # Optional: auto-detected
}
```

**Use Cases** (auto-detected if not specified):
- `googling`: Factual questions that require web search
- `ha_command`: Home Assistant control commands
- `general`: General conversation

**Example Requests**:
```bash
# Web search
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of Assyria?", "use_case": "googling"}'

# Home Assistant command
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Add gin to the shopping list", "use_case": "ha_command"}'

# General chat (use_case auto-detected)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you help me?"}'
```

### Variation Generation
```bash
POST /variations
Content-Type: application/json

{
  "base_message": "Bubbers, lock the bloody car",
  "style": "humorous",
  "count": 3
}
```

**Styles**: `humorous`, `formal`, `casual`, `friendly`

### Webhook for Home Assistant
```bash
POST /webhook/variation
Content-Type: application/json

{
  "message": "Your automation message",
  "style": "humorous",
  "count": 1
}
```

## Integration with Home Assistant

### Using the API Gateway

The API Gateway runs on port 8080 and can be accessed from Home Assistant via:

1. **REST API**: Use the `/chat` endpoint in Home Assistant automations or scripts
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
            style: "humorous"
            count: 1
      - service: notify.mobile_app
        data:
          message: "{{ states('sensor.last_variation') }}"
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
├── compose.yml              # Docker Compose configuration
├── .env                     # Environment variables (not in git)
├── api-gateway/            # Main FastAPI service
├── ha-client/              # Home Assistant client
├── search-service/         # Web search integration
├── tools/                  # Function calling definitions
├── prompts/                # System prompts for LLM
├── automation-variations/   # Variation generator
├── knowledge-base/          # Entity registry (future)
└── hf_cache/              # Model cache
```

## Notes

- Voice input/output is handled by Home Assistant directly (Whisper, Piper, etc.)
- All LLM processing happens locally on your GPU
- Web search uses DuckDuckGo (privacy-focused, no API key required)
- The system prefers concise answers for factual queries

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
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Add gin to the shopping list", "use_case": "ha_command"}'

# Play media on speakers
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Shuffle Rush on all the speakers", "use_case": "ha_command"}'

# Set timer
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Set a timer for nine minutes", "use_case": "ha_command"}'
```

**Test Web Search**:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can dogs eat mushrooms?", "use_case": "googling"}'
```

**Test Automation Variations**:
```bash
curl -X POST http://localhost:8080/variations \
  -H "Content-Type: application/json" \
  -d '{"base_message": "Bubbers, lock the bloody car", "style": "humorous", "count": 3}'
```

## Troubleshooting

### Services Not Starting
- Check Docker: `docker --version`
- Check NVIDIA runtime: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`
- Check logs: `docker compose logs`

### Service Health Issues
- Check Docker logs: `docker compose logs api-gateway`
- Check vLLM logs: `docker compose logs vllm`
- Verify HA connection: Check API Gateway startup logs
- Test vLLM: `curl http://localhost:9000/health`
- Test API Gateway: `curl http://localhost:8080/health`

### vLLM Connection Issues
- Verify vLLM is accessible: `curl http://localhost:9000/health`
- Check API key matches in `.env` and `compose.yml`
- Verify network connectivity between containers: `docker network inspect llm_llm-network`

### Home Assistant Connection Issues
- Verify HA URL is correct and accessible
- Test HA API directly: `curl -H "Authorization: Bearer YOUR_TOKEN" http://HA_URL/api/config`
- Check access token has required permissions
- Verify network connectivity (same network/subnet)

### GPU Memory Issues
If vLLM fails to start with out of memory errors:
- Check GPU memory: `nvidia-smi`
- Review GPU memory settings in `compose.yml` (currently set to 70% utilization)
- See PROGRESS.md for detailed GPU memory configuration notes
