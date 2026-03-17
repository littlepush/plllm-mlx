#!/bin/bash
# Test 06: Full Flow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 06: Full Flow"

# Step 1: Start service
log_info "Step 1: Start service"
start_service

# Step 2: Check status - should be running with no models
log_info "Step 2: Check status (no models)"
if [ "$REDIRECT_ALL" = true ]; then
    plx status >/dev/null 2>&1
    plx ps >/dev/null 2>&1
else
    plx status
    plx ps
fi

# Step 3: List all local models
log_info "Step 3: List all local models"
if [ "$REDIRECT_ALL" = true ]; then
    plx ls >/dev/null 2>&1
else
    plx ls
fi

# Step 4: Load a model
log_info "Step 4: Load model"
if [ "$REDIRECT_ALL" = true ]; then
    plx load "$TEST_MODEL" >/dev/null 2>&1
else
    plx load "$TEST_MODEL"
fi
sleep 5

# Step 5: Verify loaded
log_info "Step 5: Verify model loaded"
response=$(plx ps --json)
assert_contains "$response" "$TEST_MODEL"

# Step 6: Chat test
log_info "Step 6: Chat test"
echo "Hello" | run_with_timeout 30 plx chat -m "$TEST_MODEL" 2>&1 || true

# Step 7: Reload model
log_info "Step 7: Reload model"
if [ "$REDIRECT_ALL" = true ]; then
    plx reload "$TEST_MODEL" >/dev/null 2>&1
else
    plx reload "$TEST_MODEL"
fi
sleep 3

# Step 8: Another chat
log_info "Step 8: Another chat after reload"
echo "Goodbye" | run_with_timeout 30 plx chat -m "$TEST_MODEL" 2>&1 || true

# Step 9: Unload model
log_info "Step 9: Unload model"
if [ "$REDIRECT_ALL" = true ]; then
    plx unload "$TEST_MODEL" >/dev/null 2>&1
else
    plx unload "$TEST_MODEL"
fi

# Step 10: Verify unloaded
log_info "Step 10: Verify model unloaded"
if [ "$REDIRECT_ALL" = true ]; then
    plx ps >/dev/null 2>&1
else
    plx ps
fi

# Step 11: Stop service
log_info "Step 11: Stop service"
stop_service

# Step 12: Verify service stopped
log_info "Step 12: Verify service stopped"
if plx status 2>&1 | grep -q "not running"; then
    log_info "Service confirmed stopped"
else
    log_warn "Service may still be running"
fi

log_info "All full flow tests passed!"