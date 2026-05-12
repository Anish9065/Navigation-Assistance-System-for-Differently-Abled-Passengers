import pyttsx3
from gtts import gTTS
import os
import threading
import time

class AudioGuidance:
    def __init__(self, use_gtts=False, language='en'):
        """
        Initializes the Audio Guidance system.
        use_gtts: If True, uses Google TTS (requires internet). If False, uses pyttsx3 (offline).
        language: 'en' for English, 'hi' for Hindi
        """
        self.use_gtts = use_gtts
        self.language = language
        self.is_speaking = False
        self.last_spoken_text = ""
        self.last_spoken_time = 0
        self.cooldown = 8.0 # 8 seconds cooldown for same message

        if not self.use_gtts:
            self.engine = pyttsx3.init()
            # Try to set properties (voices, rate)
            self.engine.setProperty('rate', 150)

    def _speak_pyttsx3(self, text):
        self.is_speaking = True
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Audio Error (pyttsx3): {e}")
        self.is_speaking = False

    def _speak_gtts(self, text):
        self.is_speaking = True
        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            filename = "temp_audio.mp3"
            tts.save(filename)
            # This is a simple blocking play, requires a command-line player.
            # Using mpg123 or similar depending on OS. For simplicity we just print.
            # os.system(f"mpg123 {filename}")
            print(f"[gTTS Audio Playing]: {text}")
            time.sleep(2) # Mock playback time
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"Audio Error (gTTS): {e}")
        self.is_speaking = False

    def announce(self, text):
        """
        Announces the text. Prevents overlapping speech and enforces cooldown.
        """
        if self.is_speaking:
            return

        current_time = time.time()
        if text == self.last_spoken_text and (current_time - self.last_spoken_time) < self.cooldown:
            return # Cooldown active for the same message

        self.last_spoken_text = text
        self.last_spoken_time = current_time

        if self.use_gtts:
            thread = threading.Thread(target=self._speak_gtts, args=(text,))
        else:
            thread = threading.Thread(target=self._speak_pyttsx3, args=(text,))

        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    audio = AudioGuidance(use_gtts=False)
    audio.announce("System initialized.")
    time.sleep(2)
