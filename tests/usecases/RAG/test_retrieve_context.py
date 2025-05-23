import pytest

from domain import ChatQueryDTO, combine_langchain_docs
from infrastructure import retriever
from usecases.RAG._retrieve_context import retrieve_context

class DummyDoc:
    def __init__(self, content: str):
        self.page_content = content


def test_retrieve_context_returns_combined_context(monkeypatch):
    # Arrange: create a sample ChatQueryDTO
    sample_query = ChatQueryDTO(id_query=1, content_query='example query')
    # Dummy documents returned by the retriever
    dummy_docs = [DummyDoc('doc1'), DummyDoc('doc2')]

    # Stub out retriever.invoke to return our dummy documents
    monkeypatch.setattr(retriever, 'invoke', lambda query_str: dummy_docs)
    # Stub out combine_langchain_docs to concatenate with a marker
    monkeypatch.setattr(
        'domain.combine_langchain_docs',
        lambda docs: 'COMBINED:' + '|'.join(doc.page_content for doc in docs)
    )

    # Act: call retrieve_context
    result = retrieve_context(sample_query)

    # Assert: should return the stubbed combined context
    assert result == 'COMBINED:doc1|doc2'


def test_retrieve_context_passes_query_content_to_retriever(monkeypatch):
    sample_query = ChatQueryDTO(id_query=2, content_query='test content')
    captured = {}

    def fake_invoke(q):
        captured['received'] = q
        return []

    monkeypatch.setattr(retriever, 'invoke', fake_invoke)
    # Combine returns empty string for no docs
    monkeypatch.setattr(
        'domain.combine_langchain_docs',
        lambda docs: ''
    )

    # Act
    result = retrieve_context(sample_query)

    # Assert that retriever.invoke was called with the query content
    assert captured.get('received') == 'test content'
    # And that the result is an empty string when no docs
    assert result == ''


def test_retrieve_context_uses_domain_dto(monkeypatch):
    # Ensure that passing a non-ChatQueryDTO raises an AttributeError or TypeError
    with pytest.raises(Exception):
        # Here, missing the required .content_query attribute
        retrieve_context(None)  # type: ignore


def test_retrieve_context():
    query = "What is the meaning of life?"
    k = 3  # Number of results to retrieve
    
    results = retrieve_context(query, k)
    
    assert isinstance(results, list)
    assert len(results) <= k  # May be less than k if not enough documents exist
    
    if results:  # If any results were found
        for result in results:
            assert isinstance(result, dict)
            assert "page_content" in result
            assert "metadata" in result
            assert isinstance(result["page_content"], str)
            assert isinstance(result["metadata"], dict)


def test_retrieve_context_empty_query():
    results = retrieve_context("", k=1)
    assert isinstance(results, list)
    assert len(results) == 0


def test_retrieve_context_invalid_k():
    with pytest.raises(ValueError):
        retrieve_context("test query", k=0)
    
    with pytest.raises(ValueError):
        retrieve_context("test query", k=-1)
