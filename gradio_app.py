from brain import encoded_image,analyze_image
from voice_input import record_audio,trancribe_audio
from Ai_voice import text_to_speech,text_to_speech_elevenlabs
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

system_prompt="""You are a professional Doctor! What's in the image?Do you see any medical conditions or abnormalities? If so, please describe them in detail.
Also, provide possible diagnoses, treatment options, and recommendations for further medical evaluation if necessary.
Make sure to use medical terminology and provide accurate information based on the visual content of the image.
If the image does not contain any medical content, please state that no medical analysis is possible.
Please keep the response concise and relevant to the medical aspects of the image.
make sure to use common language and avoid technical jargon.
Keep your response concise and to the point.
No preamble,Start your answer right away.


"""

def process_input(audio_file_path,image_filepath):
    speech_to_text_output=trancribe_audio(GROQ_API_KEY=os.getenv("GROQ_API_KEY"),audio_file_path=audio_file_path,stt_model="whisper-large-v3")
    if image_filepath:
        Ai_response=analyze_image(query=system_prompt+speech_to_text_output,encoded_image=encoded_image(image_filepath),model="meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        Ai_response="Please provide an image for analysis."
    
    Ai_voice=text_to_speech_elevenlabs(Ai_response,"final.mp3")
    return speech_to_text_output,Ai_response,Ai_voice


iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Record your voice"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to text", lines=2, max_lines=4),   # small box, no scroll
        gr.Textbox(label="Response from Medibot", lines=10, max_lines=20),  # large box, expands with response
        gr.Audio(type="filepath", label="MediBot Voice")
    ],
    title="Medibot (0.2) Voice and Image Input Interface"
)

iface.launch(debug=True)
