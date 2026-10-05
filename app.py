"""
Streamlit frontend for the Ayanna RAG chatbot.

Reuses retrieve.py (chunk index + similarity search) and generate.py
(system prompt + Groq client setup) rather than duplicating that logic.
"""

import streamlit as st
from dotenv import load_dotenv
import os

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from retrieve import load_index, retrieve
from generate import SYSTEM_PROMPT, build_prompt, MODEL_NAME, EMBED_MODEL_NAME

load_dotenv()

st.title("Ask about Ayanna")


@st.cache_resource
def get_index():
    return load_index()


@st.cache_resource
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


chunks, vectors = get_index()
embed_model = get_embed_model()
client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about Ayanna...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve(question, embed_model, chunks, vectors, top_k=3)

            with st.expander("Retrieved chunks (debug info)"):
                for r in retrieved:
                    st.write(f"({r['score']:.3f}) {r['breadcrumb']}")

            user_prompt = build_prompt(question, retrieved)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content

        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
