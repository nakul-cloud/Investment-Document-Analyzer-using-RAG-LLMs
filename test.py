import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    print("❌ GEMINI_API_KEY is not set in .env")
    raise SystemExit

print(f"Using model: {model_name}")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name)

try:
    response = model.generate_content("Say: 'Hello from Gemini, everything works.'")
    print("✅ Gemini response:")
    print(response.text)
except Exception as e:
    print("❌ Gemini call failed:")
    print(e)
