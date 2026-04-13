#!/bin/bash
# Backward-compatible wrapper. Use call_external_tool.sh directly for external skills/tools.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/call_external_tool.sh" "$@"
