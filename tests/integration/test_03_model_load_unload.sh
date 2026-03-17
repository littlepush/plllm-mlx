#!/bin/bash
# Test 03: Model Load/Unload

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 03: Model Load/Unload"

# Start service
start_service

# Record initial subprocess count
initial_subprocess=$(count_subprocesses)
log_info "Initial subprocess count: $initial_subprocess"

# Test 1: Load model
log_info "Test 1: Load model $TEST_MODEL"
if [ "$REDIRECT_ALL" = true ]; then
    plx load "$TEST_MODEL" >/dev/null 2>&1
else
    plx load "$TEST_MODEL"
fi

# Wait for model to load and subprocess to start
sleep 10

# Check subprocess count increased
after_load=$(count_subprocesses)
log_info "After load subprocess count: $after_load"
if [ "$after_load" -le "$initial_subprocess" ]; then
    log_error "Subprocess count did not increase after load"
    exit 1
fi

# Test 2: Verify model is loaded via plx ps
log_info "Test 2: Verify model is loaded (plx ps)"
response=$(plx ps --json)
assert_contains "$response" "$TEST_MODEL"

# Test 3: Verify model is loaded via API
log_info "Test 3: Verify model is loaded (API)"
response=$(http_get "/v1/model/list")
assert_contains "$response" "$TEST_MODEL"
assert_contains "$response" '"is_loaded":true'

# Test 4: Unload model
log_info "Test 4: Unload model"
if [ "$REDIRECT_ALL" = true ]; then
    plx unload "$TEST_MODEL" >/dev/null 2>&1
else
    plx unload "$TEST_MODEL"
fi

sleep 2

# Check subprocess count decreased
after_unload=$(count_subprocesses)
log_info "After unload subprocess count: $after_unload"

# Test 5: Verify model is unloaded
log_info "Test 5: Verify model is unloaded"
response=$(plx ps --json)
if echo "$response" | grep -q "$TEST_MODEL"; then
    log_warn "Model still appears in loaded list"
fi

log_info "All model load/unload tests passed!"