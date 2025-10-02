import os
from dotenv import load_dotenv
import base64
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# image_path="burn.jpg"


def encoded_image(image_path):
    image_file=open(image_path,"rb")
    return  base64.b64encode(image_file.read()).decode('utf-8')
    


client = Groq(api_key=GROQ_API_KEY)
# model = "meta-llama/llama-4-scout-17b-16e-instruct"
# query="What is the diagnosis for this patient?"
def analyze_image(model,query,encoded_image):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return (response.choices[0].message.content)