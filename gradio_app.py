from brain import encoded_image, analyze_image
from voice_input import record_audio, trancribe_audio
from Ai_voice import text_to_speech, text_to_speech_elevenlabs
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

system_prompt = """You are a professional Doctor! What's in the image?Do you see any medical conditions or abnormalities? If so, please describe them in detail.
Also, provide possible diagnoses, treatment options, and recommendations for further medical evaluation if necessary.
Make sure to use medical terminology and provide accurate information based on the visual content of the image.
If the image does not contain any medical content, please state that no medical analysis is possible.
Please keep the response concise and relevant to the medical aspects of the image.
make sure to use common language and avoid technical jargon.
Keep your response concise and to the point.
No preamble,Start your answer right away.
"""

# ✅ Safe wrapper for TTS with fallback
def safe_tts(input_text, output_file="final.mp3"):
    try:
        # Try ElevenLabs first
        return text_to_speech_elevenlabs(input_text, output_file)
    except Exception as e:
        print(f"⚠️ ElevenLabs failed, falling back to gTTS. Error: {e}")
        return text_to_speech(input_text, output_file)


def process_input(audio_file_path, image_filepath):
    # Avoid sending empty audio
    if not audio_file_path or os.path.getsize(audio_file_path) == 0:
        return "No valid audio provided.", "Please record again.", None

    speech_to_text_output = trancribe_audio(
        GROQ_API_KEY=os.getenv("GROQ_API_KEY"),
        audio_file_path=audio_file_path,
        stt_model="whisper-large-v3"
    )

    if image_filepath:
        Ai_response = analyze_image(
            query=system_prompt + speech_to_text_output,
            encoded_image=encoded_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        Ai_response = "Please provide an image for analysis."

    # ✅ Use fallback-enabled TTS
    Ai_voice = safe_tts(Ai_response, "final.mp3")

    return speech_to_text_output, Ai_response, Ai_voice


with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="pink")) as iface:
    # Centered title
    gr.HTML(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h1>🩺🤖 MediBot 0.2</h1>
            <h3>A smarter voice + vision medical assistant</h3>
        </div>
        <hr>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(type="filepath", label="🎤 Record your voice")
            image_in = gr.Image(type="filepath", label="🖼️ Upload an image")

        with gr.Column(scale=2):
            speech_out = gr.Textbox(
                label="📝 Speech to text",
                lines=2,
                max_lines=4,
                placeholder="Your speech transcription will appear here..."
            )
            response_out = gr.Textbox(
                label="💡 Response from MediBot",
                lines=12,
                max_lines=20,
                placeholder="MediBot's medical insights will appear here..."
            )
            voice_out = gr.Audio(type="filepath", label="🔊 MediBot Voice", interactive=False)

            # ✅ Submit button
            submit_btn = gr.Button("🚀 Submit")

    gr.Markdown(
        """
        ---
        ⚡ Powered by **Groq** (Brain) | **Gradio** (UI) | **GTTS & ElevenLabs** (Voice)  
        """
    )

    # Wire up the button to trigger the pipeline
    submit_btn.click(
        fn=process_input,
        inputs=[audio_in, image_in],
        outputs=[speech_out, response_out, voice_out]
    )

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        debug=True
    )

