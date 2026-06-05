#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Blitztext Setup ==="

# System-Pakete (einmalig, braucht sudo)
echo ""
echo "--- System-Pakete installieren ---"
sudo apt-get install -y \
    python3-evdev \
    python3-dev \
    libportaudio2 \
    wl-clipboard \
    libnotify-bin \
    ydotool

# input-Gruppe (einmalig)
if ! groups "$USER" | grep -qw input; then
    echo ""
    echo "--- input-Gruppe hinzufügen ---"
    sudo usermod -aG input "$USER"
    echo "WICHTIG: Neu einloggen damit die Gruppe aktiv wird!"
else
    echo "input-Gruppe: bereits gesetzt"
fi

# Virtuelle Umgebung (--system-site-packages für evdev)
echo ""
echo "--- Python-Umgebung ---"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv --system-site-packages
    echo "venv erstellt"
fi

# Python-Pakete
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet openai sounddevice numpy scipy

# Optional: lokales Whisper-Backend
if [ "${1}" = "--local" ]; then
    echo "Installiere faster-whisper (für TRANSCRIPTION_BACKEND=local)…"
    .venv/bin/pip install --quiet faster-whisper
fi

echo "Pakete installiert"

# ydotool Daemon als User-Service aktivieren (Auto-Paste)
echo ""
echo "--- ydotool Daemon aktivieren ---"
systemctl --user enable --now ydotool.service 2>/dev/null && echo "ydotool.service aktiv" || echo "ydotool.service bereits aktiv oder nicht verfügbar"

# .env aus Vorlage anlegen
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env angelegt — bitte OPENAI_API_KEY eintragen!"
fi

echo ""
echo "=== Fertig ==="
echo "1. OPENAI_API_KEY in .env eintragen"
echo "2. INPUT_DEVICE in .env setzen (leer = PipeWire Standard)"
if ! groups "$USER" | grep -qw input; then
    echo "3. Neu einloggen (input-Gruppe)"
    echo "4. Starten mit: ./run.sh"
else
    echo "3. Starten mit: ./run.sh"
fi
