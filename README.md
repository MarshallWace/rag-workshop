# RAG Workshop

A hands-on workshop for building Retrieval-Augmented Generation (RAG) systems, progressing from simple word-overlap retrieval to agentic RAG with ReAct agents.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager (handles Python installation automatically)
- OpenAI API key (see below)

## Getting your OpenAI API key

1. **Create an OpenAI account** at https://platform.openai.com/

2. **Give us your OpenAI email** so we can invite you to the organization.

3. **Accept the invitation** to the "MW Hackathon" organization.
   - Check your email for subject: "You were invited to the organization MW Hackathon on OpenAI"

4. **Generate your API key:**
   - Go to https://platform.openai.com/
   - In the top left, change the organization to **MW Hackathon**
     (You should see "MW Hackathon / RAG Workshop" at the top)
   - On the left sidebar, click **API keys** (under Organization)
   - Click **Create new secret key**
   - Fill in:
     - **Name:** `<your name>`
     - **Project:** `RAG Workshop`
     - **Permissions:** `All`
   - Click **Create secret key** and copy it to your `.env` file

## Installation

1. **Install uv** (if not already installed):

   Follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for your operating system.

   Verify the installation:
   ```bash
   uv --version
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/marshallwace/rag-workshop.git
   cd rag-workshop
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```
   This will automatically download Python 3.14 and install all dependencies.

4. **Configure environment:**
   ```bash
   cp .env.sample .env
   ```
   Then edit `.env` and add your OpenAI API key (see [Getting your OpenAI API key](#getting-your-openai-api-key) above).

5. **Verify setup:**
   ```bash
   uv run python -c "from utils import get_embedding, generate_completion; print('Setup OK')"
   ```
   If this fails, check that your `.env` file exists and contains a valid `OPENAI_API_KEY`.

## Workshop Structure

> **Note:** Exercises build on each other. Complete them in order (Step 1 → Step 2).

### Step 1: Retrieval

| Exercise | Description | Command |
|----------|-------------|---------|
| Exercise 1 | Simple word-overlap retrieval | `uv run pytest step1_retrieval/exercise_1/test_retrieval.py` |
| Exercise 2 | Embedding-based retrieval | `uv run pytest step1_retrieval/exercise_2/test_retrieval.py` |

### Step 2: Generation

| Exercise | Description | Command |
|----------|-------------|---------|
| Exercise 1 | Simple RAG (retrieve + generate) | `uv run python -m step2_generation.exercise_1.demo_rag` |
| Exercise 2 | Agentic RAG with ReAct | `uv run pytest step2_generation/exercise_2/test_retrieval.py` |

## Running Tests

Run all tests:
```bash
uv run pytest
```

Run a specific exercise:
```bash
uv run pytest step1_retrieval/exercise_1/test_retrieval.py -v
```

Use `-s` to see console output (print statements, logs, etc.):
```bash
uv run pytest step1_retrieval/exercise_1/test_retrieval.py -vs
```

## API Reference

The `utils` module provides helper functions for exercises:

```python
from utils import get_embedding, generate_completion, cosine_similarity_batch

# Get embeddings
embedding = await get_embedding("some text")  # returns list[float]

# Generate completions
answer = await generate_completion("your prompt here")

# Compute similarity scores
scores = cosine_similarity_batch(query_emb, [emb1, emb2, ...])  # returns list[float]
```
