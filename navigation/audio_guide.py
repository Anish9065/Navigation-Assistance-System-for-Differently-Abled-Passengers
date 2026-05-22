"""
AudioGuide — text-to-speech navigation announcements.
Uses pyttsx3 (offline) with gTTS (online) as fallback.
"""

import threading
import time


class AudioGuide:
    def __init__(self):
        self._engine   = None
        self._lock     = threading.Lock()
        self._last_msg = ''
        self._last_time = 0
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 140)
            self._engine.setProperty('volume', 1.0)
        except Exception:
            self._engine = None

    # ── Speak (blocking) ─────────────────────────────────────────
    def speak(self, text: str):
        if not text:
            return
        with self._lock:
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    return
                except Exception:
                    pass
            # Fallback: gTTS → play with playsound
            try:
                from gtts import gTTS
                import tempfile, os
                tts = gTTS(text=text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    tmp = f.name
                tts.save(tmp)
                try:
                    import playsound
                    playsound.playsound(tmp)
                except Exception:
                    pass
                finally:
                    os.unlink(tmp)
            except Exception:
                print(f"[AudioGuide] {text}")

    # ── Speak async (non-blocking, debounced) ─────────────────────
    def speak_async(self, text: str, debounce: float = 4.0):
        now = time.time()
        if text == self._last_msg and (now - self._last_time) < debounce:
            return
        self._last_msg  = text
        self._last_time = now
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    # ── Announce destination ──────────────────────────────────────
    def announce_destination(self, destination_name: str):
        self.speak_async(f"Setting destination to {destination_name}")

    def announce_direction(self, instruction: str):
        self.speak_async(instruction)

    def announce_arrived(self):
        self.speak_async("You have arrived at your destination.")

    def welcome(self):
        self.speak_async(
            "Welcome to the Navigation Assistance System. "
            "Please select your destination to begin."
        )
