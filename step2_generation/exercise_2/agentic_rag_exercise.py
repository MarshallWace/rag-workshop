# Run tests: uv run pytest step2_generation/exercise_2/test_retrieval.py
#
# ============================================================================
# NOTE: This exercise tests only "agentic" difficulty queries!
# ============================================================================
# Agentic RAG uses a ReAct agent that can search multiple times, following
# up on initial results to gather more context. This handles:
#   - Multi-hop queries (hub chunk → detail chunks)
#   - Complex questions requiring information synthesis
#   - Queries where a single search isn't enough
#
# The tests here are specifically the "agentic" difficulty level - these are
# designed to require multiple retrieval steps. Your agent should be able to
# pass these tests that the simpler approaches couldn't handle!
# ============================================================================

import logging

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from step1_retrieval.exercise_2 import retrieve as retrieve_embeddings
from step2_generation.exercise_1 import preprocess as preprocess_rag
from tests.test_runner import TestChunk
from utils.llm_utils import llm
from utils.types import Chunks, GenerationResult, RetrievalResult

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def preprocess(test_chunks: list[TestChunk]) -> Chunks:
    """Preprocess chunks using embedding-based preprocessing.

    Uses the same preprocessing from Step 2 Exercise 1 (Simple RAG).
    """
    return await preprocess_rag(test_chunks)


async def retrieve(question: str, chunks: Chunks, top_k: int = 3) -> RetrievalResult:
    """Use embedding-based retrieval from Step 1 Exercise 2."""
    logger.info(question)
    return await retrieve_embeddings(question, chunks, top_k)


async def generate(question: str, chunks: Chunks) -> GenerationResult:
    """Generate answer using a ReAct agent that autonomously retrieves context.

    Create a ReAct agent with a retrieval tool. The agent decides when and how
    many times to search the knowledge base (multi-hop retrieval).

    Args:
        question: The user's question
        chunks: Preprocessed chunks dictionary

    Returns:
        GenerationResult with answer from the agent

    Tools available:
        ReActAgent - Agent that reasons and acts in a loop
        FunctionTool.from_defaults(async_fn, name, description) - Wrap a function as a tool
        retrieve(query, chunks, top_k) - Search the knowledge base
        llm - The language model instance
    """
    async def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for information relevant to the query.

        Use this tool to find facts, details, or context needed to answer a question.
        You can call this multiple times with different queries to gather more information.
        """
        result = await retrieve(query, chunks)
        contents = [chunks[cid]["content"] for cid in result.sources]
        return "\n\n".join(contents)

    search_tool = FunctionTool.from_defaults(
        async_fn=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for information relevant to a query. "
            "Returns text content from the most relevant chunks. "
            "Call multiple times with different queries to gather more context."
        ),
    )

    agent = ReActAgent(
        name="rag_agent",
        tools=[search_tool],
        llm=llm,
        verbose=False,
    )

    response = await agent.run(user_msg=question)
    return GenerationResult(answer=str(response))
