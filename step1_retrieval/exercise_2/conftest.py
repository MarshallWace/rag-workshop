"""Pytest configuration for Exercise 2."""

_SUMMARY = """
NOTE: You are NOT expected to pass all tests!

Embedding-based retrieval understands semantic similarity, so it handles
paraphrasing and synonyms much better than word-overlap.

Compare your results to Exercise 1 - you should see improvement! But this
approach still struggles with multi-hop queries requiring multiple chunks.

>>> Next: Move to Step 2 for GENERATION and AGENTIC RAG!

    uv run python -m step2_generation.exercise_1.demo_rag
"""


def pytest_terminal_summary(terminalreporter):
    """Print summary after tests complete."""
    terminalreporter.write_sep("=", "Exercise 2: Embedding-Based Retrieval")
    for line in _SUMMARY.strip().split("\n"):
        terminalreporter.write_line(line)
