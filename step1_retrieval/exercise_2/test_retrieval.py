import pytest
import pytest_asyncio

from step1_retrieval.exercise_2 import embedding_retrieval_exercise as embedding_retrieval
from step1_retrieval.exercise_2.embedding_retrieval_exercise import preprocess
from tests.test_runner import get_chunks, get_test_cases

_test_cases = get_test_cases("student_tech")


@pytest_asyncio.fixture(scope="module")
async def chunks():
    """Preprocess chunks once for all tests."""
    chunks_data = get_chunks("student_tech")
    return await preprocess(chunks_data)


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", _test_cases, ids=lambda tc: tc.query[:50])
async def test_retrieval(test_case, chunks, mocker):
    """Test that retrieve returns the expected chunks for each query."""
    spy = mocker.spy(embedding_retrieval, "retrieve")

    await embedding_retrieval.retrieve(test_case.query, chunks, top_k=3)

    all_sources = {src for r in spy.spy_return_list for src in r.sources}
    assert set(test_case.expected_chunk_ids) <= all_sources, (
        f"Expected: {test_case.expected_chunk_ids}\nGot: {list(all_sources)}"
    )
