#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Blitztext Setup ==="

# Virtuelle Umgebung erstellen
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "venv erstellt"
fi

# Pakete installieren
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet evdev sounddevice numpy scipy openai pystray Pillow

echo "Pakete installiert"

# .env anlegen falls nicht vorhanden
if [ ! -f ".env" ]; then
    cat > .env <<'EOF'
OPENAI_API_KEY=sk-...deinen-key-hier-eintragen...
EOF
    echo ".env Vorlage angelegt — bitte API Key eintragen!"
fi

echo ""
echo "=== Fertig ==="
echo "1. OpenAI API Key in .env eintragen"
echo "2. Neu einloggen (input-Gruppe aktiv)"
echo "3. Starten mit: ./run.sh"
