#!/bin/bash
# Test 01: Health Check

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log_section "Test 01: Health Check"

# Start service
start_service

# Test 1: Health endpoint
log_info "Test 1: GET /health"
response=$(http_get "/health")
assert_contains "$response" "healthy"
assert_contains "$response" "plllm-mlx"

# Test 2: Root endpoint
log_info "Test 2: GET /"
response=$(http_get "/")
assert_contains "$response" "plllm-mlx"
assert_contains "$response" "docs"

# Test 3: plx status
log_info "Test 3: plx status"
if [ "$REDIRECT_ALL" = true ]; then
    plx status >/dev/null 2>&1
else
    plx status
fi

# Test 4: Service is running
log_info "Test 4: Check main process"
main_count=$(count_main_processes)
if [ "$main_count" -lt 1 ]; then
    log_error "No main process found"
    exit 1
fi

log_info "All health check tests passed!"