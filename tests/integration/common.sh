#!/bin/bash
# Common functions for integration tests

set -e

# Configuration
TEST_PORT="${TEST_PORT:-18000}"
TEST_MODEL="${TEST_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
TIMEOUT="${TIMEOUT:-120}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Support QUIET mode from environment
if [ "${QUIET:-false}" = "true" ]; then
    REDIRECT_ALL=true
else
    REDIRECT_ALL=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    if [ "$REDIRECT_ALL" = false ]; then
        echo -e "${GREEN}[INFO]${NC} $1"
    fi
}

log_warn() {
    if [ "$REDIRECT_ALL" = false ]; then
        echo -e "${YELLOW}[WARN]${NC} $1"
    fi
}

log_error() {
    if [ "$REDIRECT_ALL" = false ]; then
        echo -e "${RED}[ERROR]${NC} $1"
    fi
}

log_section() {
    if [ "$REDIRECT_ALL" = false ]; then
        echo ""
        echo "========================================"
        echo "$1"
        echo "========================================"
    fi
}

# Run command with timeout (macOS compatible)
run_with_timeout() {
    local timeout_sec=$1
    shift
    # Use perl for timeout since timeout command may not exist on macOS
    perl -e 'alarm shift; exec @ARGV' "$timeout_sec" "$@"
}

# Run plx command, suppressing output in quiet mode
plx_quiet() {
    if [ "$REDIRECT_ALL" = true ]; then
        uv run plx "$@" >/dev/null 2>&1
    else
        uv run plx "$@"
    fi
}

# Check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found"
        return 1
    fi
    return 0
}

# Check if a port is open
check_port_open() {
    local port=$1
    python3 -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
result = sock.connect_ex(('localhost', $port))
sock.close()
exit(0 if result == 0 else 1)
" 2>/dev/null
}

# Wait for service to be ready
wait_for_service() {
    local port=$1
    local timeout=$2
    local count=0
    
    log_info "Waiting for service on port $port..."
    
    while [ $count -lt $timeout ]; do
        if check_port_open "$port"; then
            log_info "Service is ready on port $port"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    log_error "Timeout waiting for service on port $port"
    return 1
}

# Start the service
start_service() {
    log_info "Starting plx service on port $TEST_PORT..."
    
    # Clean up any existing service
    stop_service || true
    
    # Start service
    if [ "$REDIRECT_ALL" = true ]; then
        uv run plx serve --port "$TEST_PORT" >/dev/null 2>&1 &
    else
        uv run plx serve --port "$TEST_PORT" 2>&1 &
    fi
    local pid=$!
    
    # Wait for service to be ready
    if ! wait_for_service "$TEST_PORT" "$TIMEOUT"; then
        log_error "Failed to start service"
        return 1
    fi
    
    log_info "Service started with PID $pid"
    echo "$pid" > "$SCRIPT_DIR/.service.pid"
    return 0
}

# Stop the service
stop_service() {
    log_info "Stopping plx service..."
    
    if [ "$REDIRECT_ALL" = true ]; then
        uv run plx stop >/dev/null 2>&1 || true
    else
        uv run plx stop 2>/dev/null || true
    fi
    
    # Kill any remaining processes
    if [ -f "$SCRIPT_DIR/.service.pid" ]; then
        local pid=$(cat "$SCRIPT_DIR/.service.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$SCRIPT_DIR/.service.pid"
    fi
    
    # Wait for port to be released
    sleep 2
    
    log_info "Service stopped"
}

# Count subprocess processes
count_subprocesses() {
    pgrep -f "subprocess/python/main.py" 2>/dev/null | wc -l | tr -d ' '
}

# Get main process count
count_main_processes() {
    pgrep -f "plllm-mlx run-server" 2>/dev/null | wc -l | tr -d ' '
}

# Make HTTP request to the service
http_get() {
    local path=$1
    curl -s "http://localhost:$TEST_PORT$path"
}

http_post() {
    local path=$1
    local data=$2
    curl -s -X POST "http://localhost:$TEST_PORT$path" \
        -H "Content-Type: application/json" \
        -d "$data"
}

# Run plx command
plx() {
    uv run plx "$@"
}

# Assert that two values are equal
assert_eq() {
    if [ "$1" != "$2" ]; then
        log_error "Assertion failed: '$1' != '$2'"
        return 1
    fi
    return 0
}

# Assert that a value contains a substring
assert_contains() {
    if [[ "$1" != *"$2"* ]]; then
        log_error "Assertion failed: '$1' does not contain '$2'"
        return 1
    fi
    return 0
}

# Assert that a value is not empty
assert_not_empty() {
    if [ -z "$1" ]; then
        log_error "Assertion failed: value is empty"
        return 1
    fi
    return 0
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    stop_service
}

# Register cleanup on exit
trap cleanup EXIT