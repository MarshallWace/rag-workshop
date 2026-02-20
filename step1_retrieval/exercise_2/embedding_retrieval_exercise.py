# Run tests: uv run pytest step1_retrieval/exercise_2/test_retrieval.py
#
# ============================================================================
# NOTE: You are NOT expected to pass all tests!
# ============================================================================
# Embedding-based retrieval understands semantic similarity, so it handles
# paraphrasing and synonyms much better than word-overlap. However, it still
# struggles with:
#   - Multi-hop queries requiring information from multiple chunks
#   - Complex reasoning that needs iterative retrieval
#   - "Agentic" queries where a single search isn't enough
#
# The tests include easy, medium, hard, and agentic difficulty levels.
# Expect to pass most "easy" and many "medium" tests with this approach.
#
# Compare your results to Exercise 1 - you should see improvement!
# Then move to Step 2 to add generation and see how agentic RAG handles
# the harder queries.
# ============================================================================

import asyncio

from tests.test_runner import TestChunk
from utils import cosine_similarity_batch, get_embedding
from utils.types import Chunks, RetrievalResult


async def preprocess(test_chunks: list[TestChunk]) -> Chunks:
    """Preprocess chunks by computing embeddings for each.

    For each chunk, get its embedding and store it along with the original content.

    Args:
        test_chunks: List of TestChunk objects

    Returns:
        Chunks dictionary where each entry has:
        - Key: chunk_id (str)
        - Value: dict with "content" and "embedding" keys

    API:
        embedding = await get_embedding("some text")  # returns list[float]
    """
    embeddings = await asyncio.gather(
        *[get_embedding(chunk.chunk_content) for chunk in test_chunks]
    )

    return {
        chunk.chunk_id: {"content": chunk.chunk_content, "embedding": embedding}
        for chunk, embedding in zip(test_chunks, embeddings)
    }


async def retrieve(question: str, chunks: Chunks, top_k: int = 3) -> RetrievalResult:
    """Retrieve most relevant chunks using embedding similarity.

    Embed the question, then find chunks with the most similar embeddings.

    Args:
        question: The search query
        chunks: Preprocessed chunks dictionary (with embeddings)
        top_k: Number of top results to return

    Returns:
        RetrievalResult with sources (list of chunk_ids for top_k chunks)

    API:
        embedding = await get_embedding("some text")  # returns list[float]
        scores = cosine_similarity_batch(query_emb, [emb1, emb2, ...])  # returns list[float]
    """
    query_embedding = await get_embedding(question)

    chunk_ids = list(chunks.keys())
    chunk_embeddings = [chunks[cid]["embedding"] for cid in chunk_ids]

    scores = cosine_similarity_batch(query_embedding, chunk_embeddings)

    # Pair each chunk with its score, sort by score descending, take top_k
    scored_chunks = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    top_chunk_ids = [chunk_id for chunk_id, _score in scored_chunks[:top_k]]

    return RetrievalResult(sources=top_chunk_ids)
