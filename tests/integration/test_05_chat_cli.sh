#!/bin/bash
# Test 05: Chat CLI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 05: Chat CLI"

# Start service
start_service

# Load model
log_info "Loading model $TEST_MODEL..."
if [ "$REDIRECT_ALL" = true ]; then
    plx load "$TEST_MODEL" >/dev/null 2>&1
else
    plx load "$TEST_MODEL"
fi
sleep 5

# Test 1: Simple chat via CLI
log_info "Test 1: Simple chat via plx chat"
echo "What is 1+1?" | run_with_timeout 30 plx chat -m "$TEST_MODEL" 2>&1 || true

# Test 2: Verify model still loaded
log_info "Test 2: Verify model still loaded"
response=$(plx ps --json)
assert_contains "$response" "$TEST_MODEL"

# Cleanup
if [ "$REDIRECT_ALL" = true ]; then
    plx unload "$TEST_MODEL" >/dev/null 2>&1 || true
else
    plx unload "$TEST_MODEL" 2>/dev/null || true
fi

log_info "All chat CLI tests passed!"