#!/bin/bash
# Run all code quality checks

set -e

echo "Running code quality checks..."
echo ""

echo "=== Formatting Check (black) ==="
uv run black --check .
echo "Formatting: OK"
echo ""

echo "=== Running Tests ==="
uv run pytest backend/tests/ -v
echo ""

echo "All checks passed!"
