import os
from gtts import gTTS
from dotenv import load_dotenv
import elevenlabs
from elevenlabs.client import ElevenLabs

load_dotenv()

def text_to_speech(input_text, output_file):
    """Convert text to speech using gTTS and save as MP3."""
    audio_obj = gTTS(text=input_text, lang="en", slow=False)
    audio_obj.save(output_file)
    # ⚡ Do NOT auto-play here
    return output_file


ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_elevenlabs(input_text, output_file):
    """Convert text to speech using ElevenLabs and save as MP3."""
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="UtGkRPVGyMKDEylafQFs",   # replace with your preferred voice ID
        model_id="eleven_turbo_v2",
        output_format="mp3_44100_128"
    )
    elevenlabs.save(audio, output_file)

    # ⚡ Do NOT auto-play here
    return output_file
