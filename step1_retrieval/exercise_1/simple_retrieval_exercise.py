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
    # TODO: Convert test_chunks to dictionary format
    # HINT: Use a dictionary comprehension to create {chunk_id: {"content": chunk_content}}
    raise NotImplementedError("Students need to implement preprocess()")


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
    # TODO: Implement word-overlap scoring
    # HINT: defaultdict can help with counting
    raise NotImplementedError("Students need to implement retrieve()")
