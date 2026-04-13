#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT="${AGENT:-fox}"

show_help() {
  cat <<EOF
Usage: ./planaria_run.sh [OPTIONS]

Launcher for PLANARIA_ONE engine.

Environment variables:
  AGENT           Agent workspace name (default: fox)
  UV_CACHE_DIR    uv cache directory (default: .uv-cache)

Launcher options:
  --reset-memory  Reset memory (memory.jsonl, graph_memory.jsonl) before starting
  -h, --help      Show this help message

Options (passed to planaria_core.py):
  --agent NAME    Agent name, workspace_<agent> will be used (default: default)
  --base-dir DIR  Base directory where workspace_* folders live (default: .)
  --mode MODE     Interaction mode: cli | telegram | all (default: cli)
  --init-only     Initialize workspace/config and exit
  --message TEXT   Run one-shot message and exit
  --message-file F Read one-shot message from UTF-8 text file
  --source LABEL  Message source label for one-shot mode (default: cli)
  --user-id ID    User id label for one-shot mode (default: local)

Examples:
  ./planaria_run.sh                          # Start CLI mode
  ./planaria_run.sh --reset-memory           # Reset memory, then start CLI
  ./planaria_run.sh --mode telegram          # Start Telegram mode
  AGENT=bear ./planaria_run.sh --reset-memory  # Reset & start for workspace_bear
EOF
  exit 0
}

RESET_MEMORY=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    -h|--help) show_help ;;
    --reset-memory) RESET_MEMORY=true ;;
    *) PASSTHROUGH_ARGS+=("$arg") ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is required. Install uv first."
  exit 1
fi

cd "$BASE_DIR"
if [[ -z "${UV_CACHE_DIR:-}" ]]; then
  export UV_CACHE_DIR="$BASE_DIR/.uv-cache"
fi
mkdir -p "$UV_CACHE_DIR"

if [[ "$RESET_MEMORY" == true ]]; then
  MEM_DIR="$BASE_DIR/workspace_${AGENT}/memory"
  if [[ -d "$MEM_DIR" ]]; then
    rm -f "$MEM_DIR/memory.jsonl" "$MEM_DIR/graph_memory.jsonl"
    echo "[planaria_run] memory reset for workspace_${AGENT}"
  fi
fi

exec uv run python -u "$BASE_DIR/planaria_core.py" "${PASSTHROUGH_ARGS[@]}"
