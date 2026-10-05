"""
One-off helper: ask Groq which models your API key can actually use.
Run this once to find a valid model name for generate.py.
"""

import os
from openai import OpenAI

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY environment variable is not set.")
    raise SystemExit(1)

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

models = client.models.list()
for m in models.data:
    print(m.id)
