import os

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

API_KEY = os.getenv('API_KEY')
MODEL_ID = "open-mixtral-8x7b"
client = Mistral(api_key=API_KEY)

def load_model():
    
    client = Mistral(
        api_key=API_KEY
    )

    return client

def generate_response(client, prompt, max_tokens: int =256, temperature: float =0.0) -> str:

    response = client.chat.complete(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print(f"Api key: {API_KEY}")