import os
import pytest
from langchain_core.documents import Document
from domain import (
    combine_langchain_docs,
    to_langchain_simple_metadata,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from domain._langchain import combine_docs


def test_combine_docs_empty_list():
    # Combining an empty list should yield an empty string
    result = combine_docs([])
    assert result == ''


def test_combine_docs_single_document():
    # Combining a single document should return its content unchanged
    doc = Document(page_content='Hello world')
    result = combine_docs([doc])
    assert result == 'Hello world'


def test_combine_docs_multiple_documents():
    # Combining multiple documents should join with two line separators
    sep = os.linesep + os.linesep
    docs = [Document(page_content='First'), Document(page_content='Second'), Document(page_content='Third')]
    result = combine_docs(docs)
    expected = f"First{sep}Second{sep}Third"
    assert result == expected


def test_combine_docs_preserves_content_order():
    # Order of documents in input list should be preserved
    contents = ['A', 'B', 'C', 'D']
    docs = [Document(page_content=c) for c in contents]
    result = combine_docs(docs)
    # Split the result back and compare
    parts = result.split(os.linesep + os.linesep)
    assert parts == contents


def test_message_types():
    # Test HumanMessage
    human_msg = HumanMessage(content="Hello")
    assert human_msg.content == "Hello"
    assert human_msg.type == "human"

    # Test AIMessage
    ai_msg = AIMessage(content="Hi there!")
    assert ai_msg.content == "Hi there!"
    assert ai_msg.type == "ai"

    # Test SystemMessage
    sys_msg = SystemMessage(content="System instruction")
    assert sys_msg.content == "System instruction"
    assert sys_msg.type == "system"


def test_combine_langchain_docs():
    # Create test documents with page content and metadata
    docs = [
        {"page_content": "Content 1", "metadata": {"source": "doc1", "page": 1}},
        {"page_content": "Content 2", "metadata": {"source": "doc2", "page": 2}},
    ]
    
    combined = combine_langchain_docs(docs)
    assert isinstance(combined, str)
    assert "Content 1" in combined
    assert "Content 2" in combined
    assert "doc1" in combined
    assert "doc2" in combined


def test_combine_langchain_docs_empty():
    assert combine_langchain_docs([]) == ""


def test_to_langchain_simple_metadata():
    complex_metadata = {
        "source": "test.pdf",
        "page": 1,
        "complex_field": {"nested": "value"},
        "array_field": [1, 2, 3],
        "simple_field": "value"
    }
    
    simple_metadata = to_langchain_simple_metadata(complex_metadata)
    
    # Check that complex fields are removed/simplified
    assert isinstance(simple_metadata, dict)
    assert "complex_field" not in simple_metadata
    assert "array_field" not in simple_metadata
    
    # Check that simple fields are preserved
    assert simple_metadata["source"] == "test.pdf"
    assert simple_metadata["page"] == 1
    assert simple_metadata["simple_field"] == "value"


def test_to_langchain_simple_metadata_empty():
    assert to_langchain_simple_metadata({}) == {}
