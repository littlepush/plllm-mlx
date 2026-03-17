#!/bin/bash
# Test 04: Chat API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 04: Chat API"

# Start service
start_service

# Load model first
log_info "Loading model $TEST_MODEL..."
if [ "$REDIRECT_ALL" = true ]; then
    plx load "$TEST_MODEL" >/dev/null 2>&1
else
    plx load "$TEST_MODEL"
fi
sleep 5

# Test 1: Non-streaming chat completion
log_info "Test 1: Non-streaming chat completion"
response=$(http_post "/v1/chat/completions" '{
    "model": "'"$TEST_MODEL"'",
    "messages": [{"role": "user", "content": "Say hello in one word"}],
    "max_tokens": 20
}')
assert_contains "$response" "choices"
assert_contains "$response" "content"

# Test 2: Streaming chat completion
log_info "Test 2: Streaming chat completion"
response=$(http_post "/v1/chat/completions" '{
    "model": "'"$TEST_MODEL"'",
    "messages": [{"role": "user", "content": "Count from 1 to 3"}],
    "stream": true,
    "max_tokens": 50
}' 2>&1)
assert_contains "$response" "data:"
assert_contains "$response" "chat.completion.chunk"

# Test 3: Chat with system message
log_info "Test 3: Chat with system message"
response=$(http_post "/v1/chat/completions" '{
    "model": "'"$TEST_MODEL"'",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
}')
assert_contains "$response" "choices"

# Test 4: Model not found error
log_info "Test 4: Model not found error"
response=$(http_post "/v1/chat/completions" '{
    "model": "nonexistent/model",
    "messages": [{"role": "user", "content": "Hello"}]
}')
assert_contains "$response" "error" || assert_contains "$response" "not found"

# Cleanup
if [ "$REDIRECT_ALL" = true ]; then
    plx unload "$TEST_MODEL" >/dev/null 2>&1 || true
else
    plx unload "$TEST_MODEL" 2>/dev/null || true
fi

log_info "All chat API tests passed!"