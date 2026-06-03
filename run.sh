#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Erst setup.sh ausführen!"
    exit 1
fi

exec .venv/bin/python3 blitztext.py
