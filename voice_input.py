import logging
import sounddevice as sd
import soundfile as sf  # replaces scipy + pydub
from dotenv import load_dotenv  
from groq import Groq
import os

load_dotenv()




logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def record_audio(file_path="output.wav", duration=10, samplerate=44100, channels=1):
    try:
        logging.info("Recording audio...")
        # Record audio with sounddevice
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
        sd.wait()  # Wait until recording finishes
        logging.info("Recording complete.")

        # Save directly using soundfile (no ffmpeg / temp wav needed)
        sf.write(file_path, audio, samplerate)

        logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"Error recording audio: {e}")


audio_file_path="output.wav"
record_audio(file_path=audio_file_path, duration=5)
    # Or save as FLAC/OGG (compressed, but no ffmpeg required)
    # record_audio(file_path="output.flac", duration=5)



stt_model="whisper-large-v3"
def trancribe_audio(audio_file_path,stt_model,GROQ_API_KEY):
    
    client = Groq(api_key=GROQ_API_KEY)
    audio_file=open(audio_file_path,"rb")
    transcription=client.audio.transcriptions.create(
    model=stt_model,
    file=audio_file,
    language="en"
)

    return transcription.text
