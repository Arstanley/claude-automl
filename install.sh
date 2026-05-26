#!/usr/bin/env bash
#
# AutoML skill installer
#
# What it does:
#  1. Symlinks each agent definition into ~/.claude/agents/
#     so they're discoverable by Claude Code's Agent tool.
#  2. Installs Python deps for the Web UI (uses pip, optionally a venv).
#  3. Prints next steps.
#
# Re-running is safe — symlinks are forced and pip is idempotent.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_SRC="$SKILL_DIR/agents"
AGENTS_DEST="$HOME/.claude/agents"
WEBUI_DIR="$SKILL_DIR/webui"

echo "AutoML skill install"
echo "  skill root : $SKILL_DIR"
echo "  agents src : $AGENTS_SRC"
echo "  agents dst : $AGENTS_DEST"

# ---- agents ---------------------------------------------------------------

if [[ ! -d "$AGENTS_SRC" ]]; then
  echo "ERROR: agents directory missing at $AGENTS_SRC" >&2
  exit 1
fi

mkdir -p "$AGENTS_DEST"
echo
echo "Linking agents into $AGENTS_DEST:"
for agent in "$AGENTS_SRC"/*.md; do
  [[ -f "$agent" ]] || continue
  name="$(basename "$agent")"
  ln -sf "$agent" "$AGENTS_DEST/$name"
  echo "  ✓ $name"
done

# ---- Web UI deps ----------------------------------------------------------

echo
echo "Installing Web UI dependencies (fastapi, uvicorn, pydantic)..."
if [[ -n "${AUTOML_VENV:-}" ]]; then
  echo "  using venv at $AUTOML_VENV"
  if [[ ! -d "$AUTOML_VENV" ]]; then
    python3 -m venv "$AUTOML_VENV"
  fi
  # shellcheck disable=SC1091
  source "$AUTOML_VENV/bin/activate"
fi
pip install --quiet --upgrade -r "$WEBUI_DIR/requirements.txt"
echo "  ✓ deps installed"

# ---- done -----------------------------------------------------------------

cat <<EOF

AutoML skill installed.

Next steps:
  1. In Claude Code, type:    /automl <task description>
     (or:                     /automl --yolo <task description>   for no-gate mode)
     (or:                     /automl --ui-only                   to just start the UI)

  2. Open the dashboard in your browser:    http://localhost:7860

The agents (automl-planner, automl-researcher, automl-data-engineer,
automl-trainer, automl-evaluator, automl-reporter) are symlinked into
$AGENTS_DEST and now available to Claude Code's Agent tool.

To uninstall: rm $AGENTS_DEST/automl-*.md

EOF
