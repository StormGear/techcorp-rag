# TechCorp RAG Assistant

A hands-on Retrieval-Augmented Generation (RAG) app I built to move past diagrams and actually *run* the full loop: ingest company documents, embed them, retrieve the right chunks, and generate answers that cite those sources.

The knowledge base is a fictional company, **TechCorp** — employee policies, product specs, meeting notes, and FAQs. The assistant is a Flask chat UI that answers questions only from that corpus.

## Why this project exists

I wanted a concrete answer to questions that tutorials often skip:

- What does a document become *after* chunking, and why does overlap matter?
- How is a question turned into a vector and compared to stored chunks?
- Where does the LLM actually see retrieved context, and what happens if retrieval is weak?
- How do you show users *why* the model said something (sources), not just the answer?

This repo is that experiment: a small, readable RAG system I can inspect, break, and improve.

## How RAG works here

The pipeline follows the three steps in `ChatEngine.get_response()`:

1. **Retrieve** — The question is embedded via OpenRouter using OpenAI's `text-embedding-3-small` and queried against a **Neon PostgreSQL** database with the **pgvector** extension using cosine similarity (`<=>`). Top chunks are returned (limit 3) and de-duplicated by document title.
2. **Augment** — Retrieved text is assembled into a context block and stuffed into the prompt: *“Based on the following context… answer the user’s question.”* If the context is empty or off-topic, the model is instructed to say so.
3. **Generate** — Groq hosts the chat model. The reply is returned with source metadata and a simple confidence score (mean of retrieval similarity scores). If the LLM call fails, a fallback still surfaces the top retrieved excerpts.

On first launch, `DocumentProcessor` walks `techcorp-docs/`, splits markdown into overlapping character chunks (~500 chars, 100 overlap, preferring sentence/paragraph breaks), and stores embeddings directly in Neon (`techcorp_chunks` table).

```
User question
    → embed query
    → nearest chunks in Neon (pgvector)
    → prompt = system + context + question
    → Groq completion
    → answer + source chips in the UI
```

## Project layout

```
company-rag/
├── techcorp-docs/              # source corpus
│   ├── employee-handbook/
│   ├── product-specs/
│   ├── meeting-notes/
│   └── customers-faqs/
├── rag-assistant/
│   ├── app.py                  # Flask app, /api/chat and SSE stream
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── vector_engine.py
│   │   └── chat_engine.py
│   ├── templates/chat.html
│   └── static/
├── rag-project/requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Setup

1. Copy environment variables and configure keys:

   ```bash
   cp .env.example .env
   ```

   Set `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ALLOWED_MODELS`, and your Neon `DATABASE_URL` (e.g. `postgresql://user:password@ep-xxxx.neon.tech/neondb?sslmode=require`).

2. Create a virtualenv and install dependencies (from the repo root):

   ```bash
   cd rag-assistant
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r ../rag-project/requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

   First run generates embeddings via OpenRouter and indexes `techcorp-docs/` into Neon with `pgvector`. Open [http://127.0.0.1:5252](http://127.0.0.1:5252).

`.env` is gitignored so credentials and database connection strings stay off the repo.

## Run with Docker

The Docker image is lightweight (~120 MB) and includes the Flask app, `techcorp-docs/`, and Python dependencies. All ML compute runs via cloud APIs (Groq for chat, OpenRouter for embeddings, and Neon for vectors), requiring minimal RAM (<100 MB), perfect for deployment on free tiers like Render.

### Option 1: Pull from GitHub Container Registry (Quickest)

You can run the pre-built image directly without cloning the repository or waiting for dependencies to build.

1. **Pull the image from GHCR:**

   ```bash
   docker pull ghcr.io/stormgear/techcorp-rag:v0
   ```
   
2. **Run the container:**

   Pass your environment variables via `--env-file .env`:

   ```bash
   docker run --rm --name techcorp-rag \
     -p 5252:5252 \
     --env-file .env \
     ghcr.io/stormgear/techcorp-rag:v0
   ```

   *(On Apple Silicon / ARM64 Macs, Docker Desktop will emulate AMD64 automatically, or you can pass `--platform linux/amd64` if prompted).*

3. **Access the assistant:**

   Open [http://127.0.0.1:5252](http://127.0.0.1:5252) in your browser. All vector storage lives in your Neon database, so no local volume mounts are needed!

---

### Option 2: Build and run locally

If you want to modify code or docs and test local changes:

1. Create `.env` from the example (same keys as the venv setup above):

   ```bash
   cp .env.example .env
   ```

2. From the **repository root**, build the image:

   ```bash
   docker build -t techcorp-rag:local .
   ```

   The build is very fast and lightweight since no heavy ML frameworks (like PyTorch) are needed.

3. Run the container:

   ```bash
   docker run --rm --name techcorp-rag \
     -p 5252:5252 \
     --env-file .env \
     -v techcorp-chroma:/app/rag-assistant/chroma_db \
     techcorp-rag:local
   ```

   Open [http://127.0.0.1:5252](http://127.0.0.1:5252). To start completely fresh with indexing, remove the volume (`docker volume rm techcorp-chroma`).

Alternatively, use Docker Compose from the repo root:

```bash
docker compose up --build
```

That uses [docker-compose.yml](file:///Users/papakofi/Downloads/company-rag/docker-compose.yml) (port 5252, `.env`, and a `chroma-data` volume). Stop with `Ctrl+C`, or `docker compose down` (add `-v` if you also want to remove stored vectors).

## Try asking

- “What is our remote work policy?”
- “What are the key features of CloudSync Pro?”
- “Summarize recent meeting decisions.”
- “How do we handle customer support FAQs?”

Answers should stay grounded in the markdown files. If you add a document, clear or rebuild the collection (first-run indexing only runs when the store is empty) so retrieval can see it.

## What building this taught me

- **Chunking is a product decision, not just a preprocessing step.** Too-small chunks lose policy context; too-large chunks dilute similarity. Overlap exists so a sentence that straddles a cut still appears in a retrieved window.
- **Retrieval quality gates generation.** A fluent model will still hallucinate if the wrong (or empty) chunks are in the prompt. Source chips made that visible: I could see *which* files the answer leaned on.
- **Embeddings are just vectors; the index is the memory.** Storing embeddings in Neon with pgvector allows persistence and fast vector similarity search without managing local database files or container volumes.
- **The prompt is the glue.** The system prompt defines the assistant’s job; the augmented user prompt is where retrieved documents actually enter the model. Separating those two made RAG feel less like magic.
- **Fallbacks matter.** When the API is down, returning retrieved excerpts is more honest than a generic error — the retrieval half of RAG still has value on its own.

This is a learning system, not a production knowledge base: fixed-size chunking, a single collection, no reranker, and streaming that currently tokenizes a completed answer for the UI. Those are the next knobs I would turn now that the basic RAG path is clear.
