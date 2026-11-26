"""Speech-to-Text (STT) and Text-to-Speech (TTS) engine with English and Hindi support."""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import speech_recognition as sr

try:
    from gtts import gTTS

    HAS_GTTS = True
except Exception:
    HAS_GTTS = False


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "English": {"stt_code": "en-IN", "tts_code": "en"},
    "Hindi": {"stt_code": "hi-IN", "tts_code": "hi"},
}
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB safety cap


class SpeechEngine:
    """Handles audio transcription (WAV -> text) and spoken response synthesis (text -> MP3)."""

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    @staticmethod
    def get_supported_languages() -> list[str]:
        return list(SUPPORTED_LANGUAGES.keys())

    def transcribe_audio_bytes(
        self, audio_bytes: bytes, language: str = "English"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Transcribe WAV audio bytes using SpeechRecognition; returns (transcript, error)."""
        if not audio_bytes or len(audio_bytes) < 44:
            return None, "No valid audio data received (recording is empty or too short)."
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return None, "Audio file exceeds the 10 MB upload limit."
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return None, "Unsupported audio format: please provide a valid PCM WAV recording."

        lang_cfg = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["English"])
        stt_lang = lang_cfg["stt_code"]

        try:
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data, language=stt_lang)
            if not text or not text.strip():
                return None, "Could not detect clear speech in the audio."
            return text.strip(), None
        except sr.UnknownValueError:
            return None, "Speech was not clear enough to transcribe. Please speak again or type your message."
        except sr.RequestError as exc:
            return None, f"Speech-to-Text service unavailable (check internet connection): {exc}"
        except ValueError as exc:
            return None, f"Unsupported audio format (WAV PCM expected): {exc}"
        except Exception as exc:
            return None, f"Speech recognition error: {exc}"

    def synthesize_speech(
        self, text: str, language: str = "English"
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Convert AI response text into MP3 audio bytes using gTTS; handles offline errors gracefully."""
        if not text or not text.strip():
            return None, "Cannot synthesize empty text."
        if not HAS_GTTS:
            return None, "gTTS library is not installed."

        lang_cfg = SUPPORTED_LANGUAGES.get(language, SUPPORTED_LANGUAGES["English"])
        tts_lang = lang_cfg["tts_code"]

        try:
            tts = gTTS(text=text.strip(), lang=tts_lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read(), None
        except Exception as exc:
            return None, f"Text-to-Speech unavailable (offline or gTTS error): {exc}"
