# blitztext-linux

Linux port of [blitztext-app](https://github.com/cmagnussen/blitztext-app) — push-to-talk dictation for Linux/Wayland.

**Hold a key → speak → release → text is in your clipboard.**

> The original is macOS-only (Swift/SwiftUI/CoreML). This is a complete Python rewrite for Linux.

---

## Hotkeys

| Hotkey | Mode | What it does |
|--------|------|--------------|
| **AltGr** (hold) | Blitztext+ | Transcription + AI cleanup (removes filler words, fixes grammar) |
| **Win + Left Ctrl** (hold) | Blitztext | Pure transcription, no changes |
| **Win + Left Alt** (hold) | Blitztext $%&! | Rewrites frustrated/emotional text as calm, professional message |

After releasing the key, the result is in your clipboard — press **Ctrl+V** to paste.

---

## Requirements

- Linux (Wayland or X11), tested on Ubuntu/GNOME
- Python 3.11+
- PipeWire or ALSA audio
- OpenAI API key ([platform.openai.com](https://platform.openai.com))
- User must be in the `input` group (for keyboard access via evdev)

---

## Setup

```bash
# 1. Add yourself to the input group (once — requires logout/login)
sudo usermod -aG input $USER

# 2. Install system dependencies
sudo apt install python3-evdev python3-dev libportaudio2 wl-clipboard libnotify-bin

# 3. Create virtualenv
python3 -m venv .venv --system-site-packages
.venv/bin/pip install openai sounddevice numpy scipy

# 4. Configure
cp .env.example .env
# → edit .env: set OPENAI_API_KEY and INPUT_DEVICE

# 5. Log out and back in (for input group), then:
./run.sh
```

---

## Configuration

Copy `.env.example` to `.env` and edit:

```ini
# Part of your microphone's name — empty = PipeWire default
INPUT_DEVICE=TONOR

# Transcription backend: openai (cloud) | local (offline, slower)
TRANSCRIPTION_BACKEND=openai

# Language for Whisper: de, en, fr, ... — empty = auto-detect
WHISPER_LANGUAGE=de

# Your OpenAI API key
OPENAI_API_KEY=sk-...

# LLM for cleanup/rewrite modes
LLM_MODEL=gpt-4o-mini
```

**Finding your microphone name:**
```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```
Use any part of the device name as `INPUT_DEVICE` (e.g. `TONOR`, `Yeti`, `Samson`, `webcam`).

**Laptop users:** Leave `INPUT_DEVICE` empty to use the built-in microphone via PipeWire default.

---

## Differences from the macOS original

| | macOS (original) | Linux (this port) |
|---|---|---|
| Language | Swift / SwiftUI | Python |
| Transcription | CoreML (local) | OpenAI Whisper API or faster-whisper |
| Keyboard input | Accessibility API | evdev |
| Audio | CoreAudio / AVFoundation | sounddevice / PipeWire |
| Clipboard | NSPasteboard | wl-copy (Wayland) |
| **Auto-paste** | ✓ | ✗ (not possible on GNOME Wayland) |
| Tray icon | ✓ | optional (limited on GNOME) |

**Auto-paste on Wayland:** GNOME does not allow simulating keyboard input from background processes. After dictation, press **Ctrl+V** manually to paste. On X11/KDE this restriction may not apply.

---

## Cost estimate (OpenAI)

| What | Model | Cost |
|------|-------|------|
| Transcription | whisper-1 | ~$0.006 / minute of speech |
| Cleanup / rewrite | gpt-4o-mini | ~$0.001 per use (short texts) |

A typical workday of dictation costs a few cents.

---

## Offline mode

Set `TRANSCRIPTION_BACKEND=local` to use [faster-whisper](https://github.com/SYSTRAN/faster-whisper) without an API key:

```bash
.venv/bin/pip install faster-whisper
```

Then set `WHISPER_MODEL=base` (fast) or `medium` (more accurate) in `.env`.
No internet required, but first run downloads the model (~150 MB for base).

---

## License

MIT — fork freely, contributions welcome.
