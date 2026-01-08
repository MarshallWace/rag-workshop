"""Pytest configuration for Agentic RAG."""

_SUMMARY = """
NOTE: These tests use ONLY 'agentic' difficulty queries!

Agentic RAG uses a ReAct agent that can search multiple times, following up
on initial results to gather more context.

These queries require multi-hop retrieval - finding a 'hub' chunk that
references other chunks, then searching for those details.

If your agent is working correctly, it should pass these tests that the
simpler retrieval approaches couldn't handle!

>>> Congratulations on completing the RAG Workshop!
"""


def pytest_terminal_summary(terminalreporter):
    """Print summary after tests complete."""
    terminalreporter.write_sep("=", "Exercise 2: Agentic RAG")
    for line in _SUMMARY.strip().split("\n"):
        terminalreporter.write_line(line)
