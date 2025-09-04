#!/usr/bin/env python3
"""Check available Gemini models"""

import os
import requests

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("GEMINI_API_KEY required")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    models = response.json()
    print("Available models:")
    for model in models.get('models', []):
        name = model.get('name', '')
        if 'gemini' in name.lower():
            print(f"  - {name}")
else:
    print(f"Error: {response.status_code} - {response.text}")