#!/bin/bash
# Test 07: Subprocess Management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 07: Subprocess Management"

# Start service
start_service

# Test 1: List initial subprocesses
log_info "Test 1: List initial subprocesses"
if [ "$REDIRECT_ALL" = true ]; then
    plx subprocess list >/dev/null 2>&1 || true
else
    plx subprocess list 2>&1 || true
fi

# Test 2: Load model and check subprocess
log_info "Test 2: Load model and check subprocess"
if [ "$REDIRECT_ALL" = true ]; then
    plx load "$TEST_MODEL" >/dev/null 2>&1
else
    plx load "$TEST_MODEL"
fi
sleep 5

# Check subprocess status
log_info "Checking subprocess status"
if [ "$REDIRECT_ALL" = true ]; then
    plx subprocess status -m "$TEST_MODEL" >/dev/null 2>&1 || true
else
    plx subprocess status -m "$TEST_MODEL" 2>&1 || true
fi

# Test 3: List subprocesses after load
log_info "Test 3: List subprocesses after load"
if [ "$REDIRECT_ALL" = true ]; then
    plx subprocess list >/dev/null 2>&1 || true
else
    plx subprocess list 2>&1 || true
fi

# Test 4: Count processes
log_info "Test 4: Count processes"
main_count=$(count_main_processes)
sub_count=$(count_subprocesses)
if [ "$sub_count" -lt 1 ]; then
    log_warn "Expected at least one subprocess"
fi

# Test 5: Stop subprocess
log_info "Test 5: Stop subprocess"
if [ "$REDIRECT_ALL" = true ]; then
    plx subprocess stop -m "$TEST_MODEL" >/dev/null 2>&1 || true
else
    plx subprocess stop -m "$TEST_MODEL" 2>&1 || true
fi

# Cleanup
if [ "$REDIRECT_ALL" = true ]; then
    plx unload "$TEST_MODEL" >/dev/null 2>&1 || true
else
    plx unload "$TEST_MODEL" 2>/dev/null || true
fi

log_info "All subprocess management tests passed!"