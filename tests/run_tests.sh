#!/bin/bash
# Main test runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
QUIET=false
REDIRECT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quiet|-q)
            QUIET=true
            shift
            ;;
        --redirect|-r)
            REDIRECT_FILE="$2"
            shift 2
            ;;
        --redirect=*)
            REDIRECT_FILE="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quiet, -q           Only output final summary"
            echo "  --redirect FILE       Redirect test output to FILE"
            echo "  --redirect=FILE       Same as above"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Setup redirect
if [ -n "$REDIRECT_FILE" ]; then
    REDIRECT_DIR=$(dirname "$REDIRECT_FILE")
    mkdir -p "$REDIRECT_DIR"
    exec 3>&1 4>&2
    exec >"$REDIRECT_FILE" 2>&1
    QUIET=true
fi

# Output functions
log() {
    if [ "$QUIET" = false ]; then
        echo "$@"
    fi
}

log_always() {
    if [ "$QUIET" = true ] && [ -z "$REDIRECT_FILE" ]; then
        echo "$@"
    elif [ "$QUIET" = false ]; then
        echo "$@"
    fi
}

log ""
log "========================================"
log "  plllm-mlx Test Suite"
log "========================================"
log ""

# Run unit tests
log "========================================"
log "Running Unit Tests (pytest)"
log "========================================"

if [ "$QUIET" = true ]; then
    UNIT_OUTPUT=$(uv run pytest tests/unit/ -v --cov=plllm_mlx --cov-report=term-missing --tb=short 2>&1) || true
    UNIT_RESULT=$?
else
    uv run pytest tests/unit/ -v --cov=plllm_mlx --cov-report=term-missing --tb=short
    UNIT_RESULT=$?
fi

log ""
log "Unit tests completed with exit code: $UNIT_RESULT"

# Run integration tests
log ""
log "========================================"
log "Running Integration Tests (bash)"
log "========================================"

cd "$SCRIPT_DIR/integration"

# Export QUIET for integration tests
export QUIET

INTEGRATION_RESULT=0
FAILED_TESTS=""

for test in test_*.sh; do
    if [ -f "$test" ]; then
        log ""
        log "--- Running $test ---"
        if bash "$test"; then
            log "✓ $test passed"
        else
            log "✗ $test failed"
            INTEGRATION_RESULT=1
            FAILED_TESTS="$FAILED_TESTS $test"
        fi
    fi
done

cd "$PROJECT_ROOT"

# Restore stdout/stderr if redirected
if [ -n "$REDIRECT_FILE" ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

# Print summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Unit tests: $([ $UNIT_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo "Integration tests: $([ $INTEGRATION_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

if [ $INTEGRATION_RESULT -ne 0 ]; then
    echo "Failed tests:$FAILED_TESTS"
fi

if [ -n "$REDIRECT_FILE" ]; then
    echo "Log file: $REDIRECT_FILE"
fi

echo ""
if [ $UNIT_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi