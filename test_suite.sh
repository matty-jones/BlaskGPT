#!/bin/bash
# Comprehensive test suite for Local LLM Integration

set -e

BASE_URL="http://localhost:8080"
VLLM_URL="http://localhost:9000"
API_KEY="local-dev-key"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Local LLM Integration Test Suite"
echo "=========================================="

# Test 1: vLLM Health
echo -e "\n${YELLOW}[Test 1]${NC} Checking vLLM health..."
if curl -s -f "$VLLM_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} vLLM is healthy"
else
    echo -e "${RED}✗${NC} vLLM is not responding at $VLLM_URL"
    exit 1
fi

# Test 2: API Gateway Health
echo -e "\n${YELLOW}[Test 2]${NC} Checking API Gateway health..."
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓${NC} API Gateway is healthy"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗${NC} API Gateway health check failed"
    echo "   Response: $HEALTH_RESPONSE"
    exit 1
fi

# Test 3: vLLM Chat Test
echo -e "\n${YELLOW}[Test 3]${NC} Testing vLLM direct chat..."
VLLM_RESPONSE=$(curl -s -X POST "$VLLM_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 10
  }')
if echo "$VLLM_RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✓${NC} vLLM chat is working"
    echo "   Response preview: $(echo "$VLLM_RESPONSE" | grep -o '"content":"[^"]*' | head -1)"
else
    echo -e "${RED}✗${NC} vLLM chat test failed"
    echo "   Response: $VLLM_RESPONSE"
    exit 1
fi

# Test 4: General Chat
echo -e "\n${YELLOW}[Test 4]${NC} Testing general chat endpoint..."
CHAT_RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, can you help me?",
    "use_case": "general"
  }')
if echo "$CHAT_RESPONSE" | grep -q "response"; then
    echo -e "${GREEN}✓${NC} General chat is working"
    echo "   Response: $(echo "$CHAT_RESPONSE" | grep -o '"response":"[^"]*' | head -1 | cut -d'"' -f4 | head -c 100)"
else
    echo -e "${RED}✗${NC} General chat test failed"
    echo "   Response: $CHAT_RESPONSE"
fi

# Test 5: Googling (Web Search)
echo -e "\n${YELLOW}[Test 5]${NC} Testing googling/search endpoint..."
SEARCH_RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of Assyria?",
    "use_case": "googling"
  }')
if echo "$SEARCH_RESPONSE" | grep -q "response"; then
    echo -e "${GREEN}✓${NC} Googling/search is working"
    echo "   Response preview: $(echo "$SEARCH_RESPONSE" | grep -o '"response":"[^"]*' | head -1 | cut -d'"' -f4 | head -c 150)"
else
    echo -e "${RED}✗${NC} Googling test failed"
    echo "   Response: $SEARCH_RESPONSE"
fi

# Test 6: Use Case Auto-Detection
echo -e "\n${YELLOW}[Test 6]${NC} Testing use case auto-detection..."
AUTO_RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the speed of light?"
  }')
if echo "$AUTO_RESPONSE" | grep -q "use_case"; then
    DETECTED_USE_CASE=$(echo "$AUTO_RESPONSE" | grep -o '"use_case":"[^"]*' | cut -d'"' -f4)
    echo -e "${GREEN}✓${NC} Use case auto-detection working"
    echo "   Detected use case: $DETECTED_USE_CASE"
else
    echo -e "${RED}✗${NC} Use case auto-detection failed"
fi

# Test 7: Automation Variations
echo -e "\n${YELLOW}[Test 7]${NC} Testing automation variations..."
VAR_RESPONSE=$(curl -s -X POST "$BASE_URL/variations" \
  -H "Content-Type: application/json" \
  -d '{
    "base_message": "Test message for variation",
    "style": "humorous",
    "count": 2
  }')
if echo "$VAR_RESPONSE" | grep -q "variations"; then
    echo -e "${GREEN}✓${NC} Automation variations are working"
    echo "   Variations count: $(echo "$VAR_RESPONSE" | grep -o '"variations":\[[^]]*' | grep -o ',' | wc -l)"
else
    echo -e "${RED}✗${NC} Automation variations test failed"
    echo "   Response: $VAR_RESPONSE"
fi

# Test 8: Webhook Endpoint
echo -e "\n${YELLOW}[Test 8]${NC} Testing webhook endpoint..."
WEBHOOK_RESPONSE=$(curl -s -X POST "$BASE_URL/webhook/variation" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Test webhook message",
    "style": "humorous",
    "count": 1
  }')
if echo "$WEBHOOK_RESPONSE" | grep -q "variations"; then
    echo -e "${GREEN}✓${NC} Webhook endpoint is working"
else
    echo -e "${RED}✗${NC} Webhook test failed"
    echo "   Response: $WEBHOOK_RESPONSE"
fi

echo -e "\n=========================================="
echo -e "${GREEN}Test Suite Complete${NC}"
echo "=========================================="
echo ""
echo "Note: Home Assistant command tests require:"
echo "  - Valid HA instance at configured URL"
echo "  - Valid access token"
echo "  - Actual entities in Home Assistant"
echo ""
echo "To test HA commands manually:"
echo "  curl -X POST $BASE_URL/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"Add gin to the shopping list\", \"use_case\": \"ha_command\"}'"

