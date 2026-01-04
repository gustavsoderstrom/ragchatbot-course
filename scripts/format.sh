#!/bin/bash
# Format all Python code in the project

set -e

echo "Formatting Python code with black..."
uv run black .
echo ""
echo "Formatting complete!"
