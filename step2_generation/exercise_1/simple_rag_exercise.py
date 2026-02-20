# Run demo: uv run python -m step2_generation.exercise_1.demo_rag
#
# ============================================================================
# NOTE: This exercise has no automated tests - it's a demo!
# ============================================================================
# Simple RAG retrieves context once and generates an answer. This works well
# for straightforward questions but has limitations:
#   - Single retrieval may miss relevant context
#   - No ability to refine or follow up on initial search
#   - Can't handle multi-hop reasoning
#
# Run the demo to see it in action, then move to Exercise 2 to implement
# agentic RAG that can iteratively search for better results!
# ============================================================================

from step1_retrieval.exercise_2 import preprocess as preprocess_embeddings
from step1_retrieval.exercise_2 import retrieve as retrieve_chunks
from tests.test_runner import TestChunk
from utils import generate_completion
from utils.types import Chunks, GenerationResult, RetrievalResult


async def preprocess(test_chunks: list[TestChunk]) -> Chunks:
    """Preprocess chunks using embedding-based preprocessing from Step 1.

    Args:
        test_chunks: List of TestChunk objects

    Returns:
        Chunks dictionary with embeddings
    """
    return await preprocess_embeddings(test_chunks)


async def retrieve(question: str, chunks: Chunks, top_k: int = 3) -> RetrievalResult:
    """Retrieve relevant chunks using embedding-based retrieval from Step 1.

    Args:
        question: The search query
        chunks: Preprocessed chunks dictionary (with embeddings)
        top_k: Number of top results to return

    Returns:
        RetrievalResult with retrieved chunk IDs
    """
    return await retrieve_chunks(question, chunks, top_k)


async def generate(question: str, chunks: Chunks) -> GenerationResult:
    """Retrieve context and generate an answer.

    Call retrieve() to get relevant chunks, build a prompt with that context,
    then call the LLM.

    Args:
        question: The user's question
        chunks: Preprocessed chunks dictionary

    Returns:
        GenerationResult with answer from the LLM

    API:
        answer = await generate_completion("your prompt here")
    """
    retrieval_result = await retrieve(question, chunks, top_k=3)

    context_pieces = [chunks[cid]["content"] for cid in retrieval_result.sources]
    context = "\n\n".join(context_pieces)

    prompt = (
        f"Here is relevant information:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    answer = await generate_completion(prompt)
    return GenerationResult(answer=answer)
