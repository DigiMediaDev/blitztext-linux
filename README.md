# blitztext-linux

Linux port of [blitztext-app](https://github.com/cmagnussen/blitztext-app) — push-to-talk dictation for Linux/Wayland.

**Hold a key → speak → release → text is in your clipboard.**

> The original is macOS-only (Swift/SwiftUI/CoreML). This is a complete Python rewrite for Linux.

---

## Hotkeys

| Hotkey | Mode | What it does |
|--------|------|--------------|
| **AltGr** (hold) | Blitztext+ | Transcription + AI cleanup (removes filler words, fixes grammar) |
| **Win + Leftside Ctrl** (hold) | Blitztext | Pure transcription, no changes |
| **Win + Leftside Alt** (hold) | Blitztext $%&! | Rewrites frustrated/emotional text as calm, professional message |

After releasing the key, the transcribed text is automatically pasted into the active field. No Ctrl+V needed.

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

## Hardware compatibility

This port was developed and tested on a **repurposed desktop PC running Linux** — not a modern laptop with a built-in microphone and clean audio stack. That context matters: a MacBook user installs blitztext-app, grants mic access, and is done. On Linux desktop hardware, the audio input chain depends on what you plug in, and each connection type behaves differently.

Three input types were tested during development:

### USB microphone (recommended)

Tested with a **TONOR TC30** USB condenser microphone. This is the most reliable setup on Linux: the device registers as a dedicated ALSA/PipeWire source and is identified by a stable partial name match (`INPUT_DEVICE=TONOR`). No driver setup required.

```ini
INPUT_DEVICE=TONOR   # or any substring of your device name
```

### Bluetooth headset

Tested with a Bluetooth headset over A2DP/HFP. Bluetooth audio on Linux requires additional configuration:

- PipeWire + WirePlumber must be running (standard on Ubuntu 22.04+)
- For stable headset microphone input, HFP profile must be active (PipeWire handles this automatically on pairing)
- If audio quality is poor or mic is not recognised, add to `~/.config/wireplumber/wireplumber.conf.d/51-bluez-config.conf`:

```ini
monitor.bluez.properties = {
  bluez5.enable-sbc-xq = true
  bluez5.enable-hw-volume = false
}
```

- To prevent Bluetooth autosuspend causing dropout, add to `/etc/modprobe.d/btusb.conf`:

```
options btusb enable_autosuspend=0
```

After pairing, use `INPUT_DEVICE=` with a substring of the Bluetooth device name as shown by `sounddevice.query_devices()`.

### Analog headset / gaming headset (3.5 mm / Cinch)

Planned for testing. Analog headsets connect via the mainboard's audio jack. On desktop PCs without a dedicated sound card, mic quality varies significantly by motherboard. PipeWire should detect the device automatically; `INPUT_DEVICE` can be set to a substring like `HDA` or the card name shown by:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

If front-panel and rear audio jacks are both present, make sure the correct source is selected in your system sound settings before running blitztext.

### Why INPUT_DEVICE is a substring match

Rather than requiring an exact device name (which changes between reboots on some setups), `INPUT_DEVICE` matches any device whose name *contains* the given string. This makes the config portable across USB re-enumerations and Bluetooth reconnects.

---

## About this port — how it came to be

The original [blitztext-app](https://github.com/cmagnussen/blitztext-app) by cmagnussen is a macOS application written in Swift (97% Swift, 3% Shell). It uses macOS-exclusive frameworks throughout: SwiftUI for the UI, CoreAudio and AVFoundation for audio, the macOS Accessibility API for global hotkeys, NSPasteboard for clipboard access, and WhisperKit/CoreML for on-device transcription.

None of these components exist on Linux. A port therefore cannot be a translation of the Swift source — it is a **complete rewrite in Python** that reproduces the same user-facing behaviour using Linux-native equivalents:

| Concern | macOS original | This Linux port | Why different |
|---|---|---|---|
| Language | Swift / SwiftUI | Python | Swift has no official Linux GUI runtime |
| Global hotkeys | macOS Accessibility API | evdev (kernel input subsystem) | Only way to intercept keys system-wide on Linux without a compositor extension |
| Audio recording | CoreAudio / AVFoundation | sounddevice + PipeWire/ALSA | CoreAudio is Apple-proprietary |
| Clipboard | NSPasteboard | wl-copy (Wayland) | NSPasteboard is macOS-only |
| Auto-paste | Built-in (Accessibility API) | ydotool + ydotoold daemon | GNOME Wayland forbids arbitrary keyboard injection; ydotool uses a privileged daemon as workaround |
| Transcription (cloud) | whisper-1 → OpenAI API | gpt-4o-mini-transcribe → OpenAI API | Same API, newer/faster model |
| Transcription (offline) | WhisperKit / CoreML (Apple Neural Engine) | faster-whisper (CPU, int8) | CoreML and the Neural Engine are Apple Silicon-only |
| Tray icon | Native macOS menu bar | optional via pystray | Limited on GNOME by design |

**What is identical:** the product concept, the three dictation modes (pure transcription / AI cleanup / calm rewrite), the OpenAI API endpoints, and the LLM prompt logic. The original README served as the functional specification for this port.

**What is new in this port:**
- Auto-paste via `ydotool` (the macOS version has this built-in; on Linux it requires a separate privileged daemon)
- `gpt-4o-mini-transcribe` instead of `whisper-1` (faster, more accurate, same price tier)
- Notification chaining with `--replace-id` so only one notification is visible at a time
- `setup.sh` / `run.sh` for one-command installation and daemon management

---

## Cost estimate (OpenAI)

| What | Model | Cost |
|------|-------|------|
| Transcription | gpt-4o-mini-transcribe | ~$0.003 / minute of speech |
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
