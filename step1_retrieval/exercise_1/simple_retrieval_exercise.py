# Run tests: uv run pytest step1_retrieval/exercise_1/test_retrieval.py
#
# ============================================================================
# NOTE: You are NOT expected to pass all tests!
# ============================================================================
# Word-overlap retrieval is a basic technique with limitations. It works well
# for queries that share exact words with the chunks, but struggles with:
#   - Semantic similarity (synonyms, paraphrasing)
#   - Multi-source queries requiring multiple chunks
#   - Complex reasoning questions
#
# The tests include easy, medium, hard, and agentic difficulty levels.
# Expect to pass mainly the "easy" tests with this approach.
#
# After completing this exercise, move to Exercise 2 to see how embedding-based
# retrieval improves results!
# ============================================================================

from collections import defaultdict

from tests.test_runner import TestChunk
from utils.types import Chunks, RetrievalResult


async def preprocess(test_chunks: list[TestChunk]) -> Chunks:
    """Preprocess chunks into a searchable format.

    Convert the list of TestChunk objects into a dictionary where:
    - Key: chunk_id (str)
    - Value: dict with "content" key containing chunk_content

    Args:
        test_chunks: List of TestChunk objects

    Returns:
        Chunks dictionary mapping chunk_id to chunk data

    Example:
        >>> chunks = await preprocess([TestChunk(chunk_id="1", chunk_content="hello")])
        >>> chunks["1"]["content"]
        "hello"
    """
    return {chunk.chunk_id: {"content": chunk.chunk_content} for chunk in test_chunks}


async def retrieve(question: str, chunks: Chunks, top_k: int = 3) -> RetrievalResult:
    """Retrieve most relevant chunks using word-overlap scoring.

    For each chunk, count how many words from the question appear in the chunk.
    Return the top_k chunks with the highest scores.

    Args:
        question: The search query
        chunks: Preprocessed chunks dictionary
        top_k: Number of top results to return

    Returns:
        RetrievalResult with:
        - sources: List of chunk_ids for top_k chunks
        - metadata: Empty dict (not used in this exercise)
    """
    question_words = set(question.lower().split())

    scores = defaultdict(int)
    for chunk_id, chunk_data in chunks.items():
        chunk_words = chunk_data["content"].lower().split()
        for word in chunk_words:
            if word in question_words:
                scores[chunk_id] += 1

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_chunk_ids = [chunk_id for chunk_id, _count in ranked[:top_k]]
    return RetrievalResult(sources=top_chunk_ids)
