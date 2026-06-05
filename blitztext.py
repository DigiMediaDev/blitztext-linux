#!/usr/bin/env python3
"""
blitztext-linux: push-to-talk Diktat für Linux/Wayland

Hotkeys:
  AltGr          halten → Blitztext+   (Transkription + KI-Cleanup)   ← Standardfall
  Win + Strg L   halten → Blitztext    (reine Transkription)
  Win + Alt L    halten → Blitztext $%&! (sachlich umformulieren)

Backends:
  TRANSCRIPTION_BACKEND=realtime  → GPT-Realtime-Whisper via OpenAI SDK
  TRANSCRIPTION_BACKEND=openai    → whisper-1 REST API (Fallback)
  TRANSCRIPTION_BACKEND=local     → faster-whisper offline

Konfiguration via .env (siehe .env Vorlage)
"""

import os
import sys
import select
import signal
import time
import asyncio
import base64
import threading
import tempfile
import subprocess
import traceback
from pathlib import Path

import evdev
from evdev import ecodes
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile


SAMPLE_RATE        = 16_000
_recording_lock    = threading.Lock()
_whisper_model     = None
_noise_floor       = 300

# Hotkey-Definitionen
KEY_ALTGR   = ecodes.KEY_RIGHTALT
KEY_WIN     = ecodes.KEY_LEFTMETA
KEY_LCTRL   = ecodes.KEY_LEFTCTRL
KEY_LALT    = ecodes.KEY_LEFTALT

HALLUCINATIONS = [
    "amara.org", "untertitel der", "subtitles by", "transcribed by",
    "♪", "www.", ".com", "copyright",
]

# ── Konfiguration ─────────────────────────────────────────────────────────────

def load_env() -> dict[str, str]:
    cfg: dict[str, str] = {}
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    for key in list(cfg) + ["OPENAI_API_KEY", "OPENROUTER_API_KEY",
                             "TRANSCRIPTION_BACKEND", "WHISPER_MODEL",
                             "WHISPER_LANGUAGE", "LLM_BACKEND", "LLM_MODEL",
                             "INPUT_DEVICE"]:
        if key in os.environ:
            cfg[key] = os.environ[key]
    return cfg


# ── Kalibrierung ──────────────────────────────────────────────────────────────

def calibrate_noise(device, seconds: float = 1.5) -> float:
    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=device,
                        callback=lambda d, f, t, s: frames.append(d.copy())):
        time.sleep(seconds)
    audio = np.concatenate(frames)
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


# ── Realtime Session (OpenAI SDK) ─────────────────────────────────────────────

class RealtimeSession:
    """Streamt Audio live zur GPT-Realtime-Whisper API via OpenAI SDK."""

    def __init__(self, api_key: str, language: str | None = None):
        self._api_key   = api_key
        self._language  = language
        self._loop      = asyncio.new_event_loop()
        self._conn      = None
        self._stream    = None
        self._active    = False
        self._connected = threading.Event()
        self._done      = threading.Event()
        self._transcript: str | None = None
        self._error: str | None = None
        self._rms_vals: list[float] = []

    def start(self, device=None) -> bool:
        self._active = True
        threading.Thread(target=self._run_loop, daemon=True).start()
        if not self._connected.wait(timeout=8):
            self._error = "Verbindungs-Timeout zur Realtime API"
            return False
        if self._error:
            return False
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                device=device, callback=self._audio_cb,
            )
            self._stream.start()
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def _audio_cb(self, indata: np.ndarray, frames, t, status):
        if not self._active:
            return
        chunk = indata.copy()
        self._rms_vals.append(float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))))
        if self._conn and self._loop:
            asyncio.run_coroutine_threadsafe(self._send_chunk(chunk), self._loop)

    async def _send_chunk(self, chunk: np.ndarray):
        try:
            await self._conn.input_audio_buffer.append(
                audio=base64.b64encode(chunk.tobytes()).decode()
            )
        except Exception:
            pass

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._sdk_session())
        except Exception as e:
            self._error = str(e)
            print(f"Realtime Verbindungsfehler: {e}")
            self._connected.set()
            self._done.set()

    async def _sdk_session(self):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self._api_key)
        transcription_cfg: dict = {"model": "whisper-1"}
        if self._language:
            transcription_cfg["language"] = self._language
        try:
            async with client.beta.realtime.connect(
                model="gpt-4o-realtime-preview"
            ) as conn:
                self._conn = conn
                await conn.session.update(session={
                    "modalities": ["text"],
                    "input_audio_transcription": transcription_cfg,
                    "turn_detection": None,
                })
                self._connected.set()
                async for event in conn:
                    kind = event.type
                    if kind == "conversation.item.input_audio_transcription.completed":
                        self._transcript = event.transcript.strip()
                        self._done.set()
                        return
                    elif kind == "error":
                        self._error = str(getattr(event, "error", event))
                        print(f"Realtime Fehler: {self._error}")
                        self._done.set()
                        return
        except Exception as e:
            self._error = str(e)
            self._connected.set()
            self._done.set()

    def stop(self) -> tuple[str | None, float]:
        self._active = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        avg_rms = float(np.mean(self._rms_vals)) if self._rms_vals else 0.0
        if self._conn and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._conn.input_audio_buffer.commit(), self._loop
            )
        self._done.wait(timeout=15)
        if self._error:
            raise RuntimeError(self._error)
        return self._transcript, avg_rms


# ── Standard-Recorder (openai REST / local) ───────────────────────────────────

class Recorder:
    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._active = False
        self._stream: sd.InputStream | None = None

    def start(self, device=None):
        with _recording_lock:
            if self._active:
                return
            self._frames = []
            self._active = True
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                device=device, callback=self._cb,
            )
            self._stream.start()

    def _cb(self, indata, frames, t, status):
        if self._active:
            self._frames.append(indata.copy())

    def stop(self) -> np.ndarray | None:
        with _recording_lock:
            if not self._active:
                return None
            self._active = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        if not self._frames:
            return None
        return np.concatenate(self._frames, axis=0)


# ── Transkription ─────────────────────────────────────────────────────────────

def transcribe_openai(audio: np.ndarray, cfg: dict) -> str:
    from openai import OpenAI
    key = cfg.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY fehlt in .env")
    client = OpenAI(api_key=key)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        wavfile.write(tmp.name, SAMPLE_RATE, audio)
        with open(tmp.name, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text",
                language=cfg.get("WHISPER_LANGUAGE") or None,
            )
        return str(result).strip()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def transcribe_local(audio: np.ndarray, cfg: dict) -> str:
    global _whisper_model
    from faster_whisper import WhisperModel
    model_size = cfg.get("WHISPER_MODEL", "base")
    if _whisper_model is None:
        print(f"Lade Whisper-Modell '{model_size}'…")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Modell geladen.")
    audio_f32 = audio.flatten().astype(np.float32) / 32768.0
    segments, _ = _whisper_model.transcribe(
        audio_f32, beam_size=5, language=cfg.get("WHISPER_LANGUAGE") or None
    )
    return " ".join(s.text for s in segments).strip()


def transcribe(audio: np.ndarray, cfg: dict) -> str:
    backend = cfg.get("TRANSCRIPTION_BACKEND", "local")
    if backend == "openai":
        return transcribe_openai(audio, cfg)
    return transcribe_local(audio, cfg)


# ── LLM-Modi ─────────────────────────────────────────────────────────────────

PROMPTS = {
    "cleanup": (
        "Bereinige diesen diktierten Text: Korrigiere Grammatik, entferne Versprecher, "
        "Wiederholungen und Füllwörter. Behalte exakt den Inhalt und die Aussagen des "
        "Originals — füge nichts hinzu, lasse nichts weg. "
        "Antworte nur mit dem bereinigten Text."
    ),
    "calm": (
        "Schreib diese frustrierte oder emotionale Aussage als ruhige, sachliche "
        "Nachricht um. Behalte den Kern, entferne Ärger und Emotionen. "
        "Antworte nur mit der umgeschriebenen Nachricht."
    ),
}


def llm_rewrite(text: str, mode: str, cfg: dict) -> str:
    prompt = PROMPTS.get(mode, "")
    if not prompt:
        return text

    from openai import OpenAI
    key = cfg.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY fehlt in .env")
    client = OpenAI(api_key=key)
    model = cfg.get("LLM_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def to_clipboard(text: str):
    proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()


def notify(title: str, body: str = ""):
    subprocess.run(["notify-send", "-t", "4000", title, body], check=False)


def is_hallucination(text: str) -> bool:
    return any(h in text.lower() for h in HALLUCINATIONS)


def finish(text: str | None, mode: str, cfg: dict):
    if not text or is_hallucination(text):
        notify("Nichts erkannt", "Bitte nochmal sprechen.")
        return
    if mode in ("cleanup", "calm"):
        try:
            notify(f"{'Blitztext+' if mode == 'cleanup' else 'Blitztext $%&!'} läuft…")
            text = llm_rewrite(text, mode, cfg)
        except Exception as e:
            notify("LLM Fehler", str(e)[:100])
            return
    to_clipboard(text)
    label = {"transcribe": "Blitztext", "cleanup": "Blitztext+", "calm": "Blitztext $%&!"}
    notify(f"{label.get(mode, 'Fertig')} — Strg+V", text[:120])
    print(f"→ [{mode}] {text[:80]}")


def resolve_input_device(cfg: dict) -> int | None:
    name = cfg.get("INPUT_DEVICE", "").strip()
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            print(f"Eingabegerät: [{i}] {d['name']}")
            return i
    print(f"Eingabegerät '{name}' nicht gefunden, nutze Standard.")
    return None


def find_keyboards() -> list[str]:
    by_id = Path("/dev/input/by-id")
    if not by_id.exists():
        return []
    result = []
    for p in by_id.iterdir():
        if "event-kbd" not in p.name:
            continue
        real = str(p.resolve())
        try:
            dev = evdev.InputDevice(real)
            caps = dev.capabilities().get(ecodes.EV_KEY, [])
            is_mouse = "mouse" in dev.name.lower()
            has_keys = ecodes.KEY_RIGHTCTRL in caps or ecodes.KEY_RIGHTALT in caps
            dev.close()
            if has_keys and not is_mouse:
                result.append(real)
        except Exception:
            pass
    return result


# ── Handler ───────────────────────────────────────────────────────────────────

def handle_realtime(session: RealtimeSession, mode: str, cfg: dict):
    try:
        text, rms = session.stop()
        threshold = _noise_floor * 2.5
        print(f"Realtime: RMS={rms:.0f}  Schwelle={threshold:.0f}  Modus={mode}")
        if rms < threshold:
            notify("Zu leise", f"RMS {rms:.0f} / Schwelle {threshold:.0f}")
            return
        finish(text, mode, cfg)
    except Exception:
        print(f"--- FEHLER ---\n{traceback.format_exc()}---")
        notify("Fehler", traceback.format_exc().splitlines()[-1][:100])


def handle_transcription(audio: np.ndarray, mode: str, cfg: dict):
    try:
        text = transcribe(audio, cfg)
        finish(text, mode, cfg)
    except Exception:
        print(f"--- FEHLER ---\n{traceback.format_exc()}---")
        notify("Fehler", traceback.format_exc().splitlines()[-1][:100])


# ── Tastatur-Listener ─────────────────────────────────────────────────────────

# (code, modifier_key) → Modus wenn Win gleichzeitig gehalten
COMBOS: dict[frozenset, str] = {
    frozenset([KEY_WIN, KEY_LCTRL]): "transcribe",
    frozenset([KEY_WIN, KEY_LALT]):  "calm",
}
SINGLE_KEYS: dict[int, str] = {
    KEY_ALTGR: "cleanup",
}
TRIGGER_KEYS = {KEY_WIN, KEY_LCTRL, KEY_LALT, KEY_ALTGR}


def detect_mode(keys: set) -> str | None:
    for combo, mode in COMBOS.items():
        if combo.issubset(keys):
            return mode
    for key, mode in SINGLE_KEYS.items():
        if key in keys:
            return mode
    return None


def listen_keyboard(kbd_path: str, recorder: Recorder, cfg: dict,
                    stop_event: threading.Event, device=None):
    try:
        kbd = evdev.InputDevice(kbd_path)
    except PermissionError:
        print(f"Keine Leseberechtigung für {kbd_path}")
        stop_event.set()
        return

    print(f"Höre auf: {kbd.name}")
    backend = cfg.get("TRANSCRIPTION_BACKEND", "local")
    keys_held: set[int] = set()
    held = False
    current_mode: str | None = None
    rt_session: RealtimeSession | None = None

    try:
        while not stop_event.is_set():
            r, _, _ = select.select([kbd.fd], [], [], 0.5)
            if not r:
                continue

            for event in kbd.read():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.code not in TRIGGER_KEYS:
                    continue

                if event.value == 1:  # Taste gedrückt
                    keys_held.add(event.code)
                    if not held:
                        mode = detect_mode(keys_held)
                        if mode:
                            held = True
                            current_mode = mode
                            if backend == "realtime":
                                rt_session = RealtimeSession(
                                    api_key=cfg.get("OPENAI_API_KEY", ""),
                                    language=cfg.get("WHISPER_LANGUAGE") or None,
                                )
                                label = {"transcribe": "Blitztext", "cleanup": "Blitztext+", "calm": "Blitztext $%&!"}
                                if rt_session.start(device=device):
                                    notify(f"{label[mode]} — Streame…", "Taste halten und sprechen")
                                else:
                                    notify("Fehler", rt_session._error or "Verbindung fehlgeschlagen")
                                    rt_session = None
                                    held = False
                                    current_mode = None
                            else:
                                label = {"transcribe": "Blitztext", "cleanup": "Blitztext+", "calm": "Blitztext $%&!"}
                                notify(f"{label[mode]} — Aufnahme…", "Taste halten und sprechen")
                                recorder.start(device=device)

                elif event.value == 0:  # Taste losgelassen
                    keys_held.discard(event.code)
                    if held and event.code in TRIGGER_KEYS:
                        held = False
                        mode_done = current_mode
                        current_mode = None

                        if backend == "realtime" and rt_session:
                            notify("Warte auf Transkription…")
                            threading.Thread(
                                target=handle_realtime,
                                args=(rt_session, mode_done, cfg),
                                daemon=True,
                            ).start()
                            rt_session = None
                        else:
                            audio = recorder.stop()
                            if audio is None or len(audio) < SAMPLE_RATE // 2:
                                notify("Zu kurz", "Länger sprechen.")
                                continue
                            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
                            threshold = _noise_floor * 2.5
                            print(f"Aufnahme: RMS={rms:.0f}  Schwelle={threshold:.0f}  Modus={mode_done}")
                            if rms < threshold:
                                notify("Zu leise", f"RMS {rms:.0f} / Schwelle {threshold:.0f}")
                                continue
                            pad = np.zeros((SAMPLE_RATE * 3 // 10, 1), dtype=np.int16)
                            audio = np.concatenate([pad, audio])
                            notify(f"Transkription läuft…")
                            threading.Thread(
                                target=handle_transcription,
                                args=(audio, mode_done, cfg),
                                daemon=True,
                            ).start()
    except OSError:
        pass
    finally:
        kbd.close()


# ── Tray-Icon ─────────────────────────────────────────────────────────────────

def try_tray(stop_event: threading.Event) -> bool:
    try:
        import pystray
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([4, 4, 60, 60], fill=(30, 120, 220))
        d.text((18, 20), "BT", fill="white")
        icon = pystray.Icon(
            "blitztext", img, "Blitztext",
            menu=pystray.Menu(
                pystray.MenuItem("Blitztext", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Beenden", lambda: (stop_event.set(), icon.stop())),
            ),
        )
        icon.run()
        return True
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_env()
    backend = cfg.get("TRANSCRIPTION_BACKEND", "local")

    if backend in ("openai", "realtime") and not cfg.get("OPENAI_API_KEY"):
        print(f"TRANSCRIPTION_BACKEND={backend} aber OPENAI_API_KEY fehlt in .env")
        sys.exit(1)

    recorder = Recorder()
    stop_event = threading.Event()

    keyboards = find_keyboards()
    if not keyboards:
        print("Keine Tastatur gefunden.")
        sys.exit(1)

    input_device = resolve_input_device(cfg)

    if backend != "realtime":
        global _noise_floor
        print("Kalibriere Mikrofon (1,5 Sek. still sein)…")
        _noise_floor = calibrate_noise(input_device)
        print(f"Rauschen: {_noise_floor:.0f}  Schwelle: {_noise_floor * 2.5:.0f}")

    for kbd in keyboards:
        threading.Thread(
            target=listen_keyboard,
            args=(kbd, recorder, cfg, stop_event, input_device),
            daemon=True,
        ).start()

    print(f"\nBlitztext gestartet  [Backend: {backend}]")
    print("  AltGr          → Blitztext+ (Transkription + KI-Cleanup)")
    print("  Win + Strg L   → Blitztext  (reine Transkription)")
    print("  Win + Alt L    → Blitztext $%&! (sachlich umformulieren)")
    print("Beenden: Strg+C\n")

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    if not try_tray(stop_event):
        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)


if __name__ == "__main__":
    main()
