"""
Step 4: generation -- the final piece of the RAG pipeline.

Flow for each question:
1. Retrieve the top-k most relevant chunks (reusing retrieve.py).
2. Stuff those chunks into a prompt as "context".
3. Send that prompt to Grok (xAI's LLM) and ask it to answer using ONLY
   that context.
4. Print Grok's answer.

The key idea that makes this "RAG" rather than just "chatbot with a system
prompt": the LLM never sees your whole document. It only ever sees the
handful of chunks retrieval decided were relevant to THIS question. That's
what keeps answers grounded and lets the doc scale far beyond what would
fit in one prompt.
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from retrieve import load_index, retrieve

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "openai/gpt-oss-120b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a chatbot that answers questions about Ayanna Jack, \
speaking about her in the third person (she/her). You will be given some CONTEXT \
pulled from Ayanna's personal source document, followed by a QUESTION.

Rules:
- Only use facts that appear in the CONTEXT. Do not invent, assume, or infer \
anything beyond what is explicitly stated.
- Do not add interpretive commentary, motivations, or editorializing that isn't \
directly stated in the CONTEXT (e.g. don't turn a stated fact into a claim about \
her values, character, or intentions unless the CONTEXT says that directly).
- Report facts plainly and concisely. Do not paraphrase quotes into dramatic or \
reflective statements.
- If the CONTEXT doesn't contain enough information to answer, say so honestly \
instead of guessing.
- Pay close attention to status language in the CONTEXT such as "planned," \
"pipeline," "considering," "next," or "first confirmed portfolio project." \
Anything described this way has NOT been done yet -- describe it as an upcoming \
or planned project, using future/conditional language (e.g. "she plans to build," \
"this is on her roadmap"), never as something she has already completed.
- When an answer involves multiple distinct items (e.g. several projects, \
skills, hobbies, or events), format them as a bullet list instead of a \
run-on sentence. Use plain "-" bullets, one item per line. For a single-fact \
answer, keep it as a normal sentence -- don't force bullets where there's \
only one thing to say.
-- Whenever you mention Ayanna's LinkedIn profile, format it as a Markdown \
link using the full URL with the https:// prefix, like this: \
[Ayanna's LinkedIn](https://linkedin.com/in/ayanna-jack-b02929271). Never \
write the URL as plain unlinked text.


- Keep answers short and direct -- a few sentences, not a narrative.
"""


def build_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def main():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print('Set it first, e.g. in PowerShell: $env:GROQ_API_KEY = "your-key-here"')
        return

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    chunks, vectors = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Loaded {len(chunks)} chunks. Ready.")

    print("\nAsk a question about Ayanna (or 'quit' to exit).")
    while True:
        question = input("\n> ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        retrieved = retrieve(question, embed_model, chunks, vectors, top_k=3)

        print("\n[retrieved chunks]")
        for r in retrieved:
            print(f"  - ({r['score']:.3f}) {r['breadcrumb']}")

        user_prompt = build_prompt(question, retrieved)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content
        print(f"\n[answer]\n{answer}")


if __name__ == "__main__":
    main()
