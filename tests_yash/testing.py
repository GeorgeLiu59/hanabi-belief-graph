from google import genai
import os

# Configure the client with your API key (ensure you have it in an environment variable)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)

print("Available Models:")
try:
    for model in client.models.list():
        print(f"* Name: {model.name}")
        print(f"  Description: {model.description}")
        print(f"  Supported methods: {model.supported_generation_methods}\n")
except Exception as e:
    print(f"Error listing models: {e}")
