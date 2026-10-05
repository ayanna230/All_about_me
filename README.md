# All About Me: A Personal RAG Chatbot

A Retrieval-Augmented Generation (RAG) app that answers questions about me, Ayanna Jack, using a personal source document as its knowledge base. Instead of relying on what a language model already "knows", it looks up the most relevant parts of my document first and uses them to write grounded answers.

## How it works

```
source document → chunks → embeddings → retrieval → generation → app
```

1. **Chunk:** `chunker.py` splits `ayanna_jack_source_document.md` into smaller passages and saves them to `chunks.json`.
2. **Embed:** `embed.py` converts each chunk into a vector and saves them to `embeddings.json`.
3. **Retrieve:** `retrieve.py` embeds the user's question and finds the chunks most similar to it.
4. **Generate:** `generate.py` sends the question plus the retrieved chunks to the language model, which writes the answer.
5. **Serve:** `app.py` ties it all together in a user interface.

## Project structure

| File | Purpose |
|---|---|
| `ayanna_jack_source_document.md` | The knowledge base the chatbot answers from |
| `chunker.py` | Splits the source document into chunks |
| `chunks.json` | Output of the chunking step |
| `embed.py` | Creates embeddings for each chunk |
| `embeddings.json` | Stored vectors for the chunks |
| `requirements.txt` | Python dependencies |
| `retrieve.py` | Finds the most relevant chunks for a question |
| `generate.py` | Builds the prompt and gets the model's answer |
| `app.py` | The app entry point |

## Getting started

### Prerequisites

- Python 3.10 or newer
- A Groq API key

### Installation

```bash
git clone https://github.com/ayanna230/All_about_me.git
cd All_about_me
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Never commit this file. It is listed in `.gitignore`.

### Run it

If you change the source document, rebuild the data first:

```bash
python chunker.py
python embed.py
```

Then start the app:

```bash
python app.py
```

## Example questions

- What am I studying, and where?
- What projects have I worked on?
- What are my technical skills?

## Tech stack

- Python
- Sentence Transformers and NumPy for semantic search
- Groq API for answer generation
- Streamlit

## Roadmap

- [ ] Add a vector database for faster retrieval
- [ ] Show source chunks alongside each answer
- [ ] Deploy the app online

## Author

**Ayanna Jack**
[LinkedIn](https://linkedin.com/in/ayanna-jack-b02929271) · [GitHub](https://github.com/ayanna230)