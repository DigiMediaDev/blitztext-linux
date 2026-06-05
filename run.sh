#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Erst setup.sh ausführen!"
    exit 1
fi

# ydotool Daemon starten falls nicht aktiv (für Auto-Paste)
if ! systemctl --user is-active --quiet ydotool.service 2>/dev/null; then
    systemctl --user start ydotool.service 2>/dev/null || true
fi

exec .venv/bin/python3 blitztext.py
