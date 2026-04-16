#!/usr/bin/env bash
set -euo pipefail

# ── PLANARIA_ONE installer ──
# Usage: curl -sSL https://raw.githubusercontent.com/jkim8k/planaria-deploy/main/install.sh | bash

REPO="https://github.com/jkim8k/planaria-deploy.git"
BRANCH="main"
INSTALL_DIR="${PLANARIA_DIR:-$HOME/planaria}"

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║  PLANARIA_ONE  —  Agent Engine    ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

# ── 1. Check git ──
if ! command -v git >/dev/null 2>&1; then
  echo "[error] git is required. Install git first."
  exit 1
fi

# ── 2. Install uv if missing ──
if ! command -v uv >/dev/null 2>&1; then
  echo "[install] uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 3. Clone ──
if [ -d "$INSTALL_DIR" ]; then
  echo "[install] $INSTALL_DIR already exists — pulling latest..."
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
else
  echo "[install] cloning into $INSTALL_DIR ..."
  git clone --branch "$BRANCH" --single-branch --depth 1 "$REPO" "$INSTALL_DIR"
fi

# ── 4. Setup ──
cd "$INSTALL_DIR"
uv sync --quiet

# ── 5. Create 'planaria' command ──
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/planaria" <<'WRAPPER'
#!/usr/bin/env bash
PLANARIA_HOME="${PLANARIA_DIR:-$HOME/planaria}"
exec "$PLANARIA_HOME/planaria_run.sh" "$@"
WRAPPER
chmod +x "$BIN_DIR/planaria"

# Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$rc" ] && ! grep -q 'planaria' "$rc"; then
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
    fi
  done
  export PATH="$BIN_DIR:$PATH"
fi

echo ""
echo "  [ok] installed successfully!"
echo ""
echo "  Start now:"
echo "    planaria"
echo ""
echo "  (onboarding will guide you through setup)"
echo ""

# ── 6. Auto-launch (reattach stdin to terminal for curl|bash) ──
exec planaria --mode cli < /dev/tty
