# src/api_interface.py

import openai
from config import OPENAI_API_KEY, GEMINI_API_KEY, MODEL_NAME

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response(prompt, model=MODEL_NAME):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content

# Function skeleton for Google Gemini 
def get_gemini_response(prompt):
    pass
