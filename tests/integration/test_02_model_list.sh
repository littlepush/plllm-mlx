#!/bin/bash
# Test 02: Model List

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 02: Model List"

# Start service
start_service

# Test 1: plx ls command
log_info "Test 1: plx ls"
if [ "$REDIRECT_ALL" = true ]; then
    plx ls >/dev/null 2>&1
else
    plx ls
fi

# Test 2: plx ls --json
log_info "Test 2: plx ls --json"
response=$(plx ls --json)
assert_contains "$response" "model_name"

# Test 3: GET /v1/model/list
log_info "Test 3: GET /v1/model/list"
response=$(http_get "/v1/model/list")
assert_contains "$response" "data"

# Test 4: GET /v1/models (OpenAI compatible)
log_info "Test 4: GET /v1/models"
response=$(http_get "/v1/models")
assert_contains "$response" "object"
assert_contains "$response" "list"

# Test 5: plx ps (loaded models)
log_info "Test 5: plx ps"
if [ "$REDIRECT_ALL" = true ]; then
    plx ps >/dev/null 2>&1
else
    plx ps
fi

# Test 6: plx ps --json
log_info "Test 6: plx ps --json"
response=$(plx ps --json)

log_info "All model list tests passed!"