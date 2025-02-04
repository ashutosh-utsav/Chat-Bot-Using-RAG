import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Get available models
models = genai.list_models()
for model in models:
    print(model.name, "->", model.supported_generation_methods)
