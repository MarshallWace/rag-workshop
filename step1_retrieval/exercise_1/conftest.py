"""Pytest configuration for Exercise 1."""

_SUMMARY = """
NOTE: You are NOT expected to pass all tests!

Word-overlap retrieval is a simple baseline. It works for queries that share
exact words with chunks, but struggles with semantic similarity (synonyms,
paraphrasing) and complex queries.

The tests include easy, medium, hard, and agentic difficulty levels.
Expect to pass mainly the 'easy' tests with this approach.

>>> Next: Run Exercise 2 to see how EMBEDDING-BASED retrieval improves results!

    uv run pytest step1_retrieval/exercise_2/test_retrieval.py -v
"""


def pytest_terminal_summary(terminalreporter):
    """Print summary after tests complete."""
    terminalreporter.write_sep("=", "Exercise 1: Word-Overlap Retrieval")
    for line in _SUMMARY.strip().split("\n"):
        terminalreporter.write_line(line)
