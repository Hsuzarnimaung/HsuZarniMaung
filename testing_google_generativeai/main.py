import google.generativeai as genai
import os

gemini_api_key = os.environ["AIzaSy"]
genai.configure(api_key = gemini_api_key)
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Who is the GOAT in the NBA?")


print(response.text)
