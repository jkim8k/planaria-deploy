#!/bin/bash
# call_external_tool.sh - External Tool Runner for planaria_one

TOOL_NAME=""
SHOW_HELP=0
TEST_MODE=0
ARGS=""
WORKSPACE="."

while [[ $# -gt 0 ]]; do
  case $1 in
    -tool)
      TOOL_NAME="$2"
      shift 2
      ;;
    -w|--workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    -test)
      TEST_MODE=1
      shift
      ;;
    *)
      # all remaining args are passed to the tool
      ARGS="$*"
      break
      ;;
  esac
done

if [ -z "$TOOL_NAME" ]; then
    echo '{"error": "-tool {name} is required."}'
    exit 1
fi

SKILL_DIR="$WORKSPACE/skills/$TOOL_NAME"

if [ ! -d "$SKILL_DIR" ]; then
    echo "{\"error\": \"Tool '${TOOL_NAME}' not found in ${WORKSPACE}/skills\"}"
    exit 1
fi

# 1. Provide schema if -h or --help is requested
if [ $SHOW_HELP -eq 1 ]; then
    MANIFEST="$SKILL_DIR/manifest.json"
    if [ -f "$MANIFEST" ]; then
        cat "$MANIFEST"
        exit 0
    else
        echo "{\"error\": \"No manifest.json found for tool: ${TOOL_NAME}\"}"
        exit 1
    fi
fi

# 2. Add some mock parameters if -test is requested
if [ $TEST_MODE -eq 1 ]; then
    # Usually in test mode we'd want to just verify it runs without crashing, or pass a dummy JSON
    ARGS='{"test_mode": true}'
    echo "[TEST MODE] Expected execution with args: $ARGS"
fi

# 3. Execute the tool
if [ -f "$SKILL_DIR/run.sh" ]; then
    bash "$SKILL_DIR/run.sh" "$ARGS"
elif [ -f "$SKILL_DIR/run.py" ]; then
    python3 "$SKILL_DIR/run.py" "$ARGS"
else
    echo "{\"error\": \"Neither run.sh nor run.py found in ${SKILL_DIR}\"}"
    exit 1
fi
