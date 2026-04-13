#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_NAME="${1:-default}"
WORKSPACE_DIR="$BASE_DIR/workspace_${AGENT_NAME}"

echo "[setup] base dir: $BASE_DIR"
echo "[setup] agent: $AGENT_NAME"

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is required"
  exit 1
fi

cd "$BASE_DIR"
uv sync

uv run python "$BASE_DIR/planaria_core.py" --agent "$AGENT_NAME" --base-dir "$BASE_DIR" --init-only

if [ ! -f "$WORKSPACE_DIR/README.md" ]; then
  cat > "$WORKSPACE_DIR/README.md" <<EOF
# workspace_${AGENT_NAME}

This is an isolated PLANARIA_ONE workspace for agent '${AGENT_NAME}'.

- Config: ./config.json
- Memory: ./memory/memory.jsonl
- Skills: ./skills/
- Logs: ./logs/
EOF
fi

echo ""
echo "[setup] workspace ready: $WORKSPACE_DIR"
echo ""

# ── Telegram setup (optional) ──
echo "Telegram 봇을 연동하시겠습니까? (y/N)"
read -r USE_TG
if [[ "${USE_TG,,}" == "y" || "${USE_TG,,}" == "yes" ]]; then
  echo "Telegram Bot Token을 입력해주세요 (@BotFather에서 발급):"
  read -r TG_TOKEN
  if [ -n "$TG_TOKEN" ]; then
    # Update config.json with telegram settings
    uv run python -c "
import json
from pathlib import Path
p = Path('$WORKSPACE_DIR/config.json')
d = json.loads(p.read_text())
d.setdefault('telegram', {})
d['telegram']['enabled'] = True
d['telegram']['token'] = '$TG_TOKEN'
d['telegram']['allow_from'] = ['*']
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
print('[setup] telegram configured successfully')
"
    echo ""
    echo "[setup] Telegram 모드로 시작:"
    echo "  ./planaria_run.sh --agent $AGENT_NAME --mode telegram"
  else
    echo "[setup] 토큰이 비어있어 Telegram 설정을 건너뜁니다."
  fi
fi

echo ""
echo "[setup] 시작 방법:"
echo "  CLI 모드:      ./planaria_run.sh --agent $AGENT_NAME --mode cli"
echo "  Telegram 모드: ./planaria_run.sh --agent $AGENT_NAME --mode telegram --telegram-token YOUR_TOKEN"
echo "  둘 다:         ./planaria_run.sh --agent $AGENT_NAME --mode all --telegram-token YOUR_TOKEN"
