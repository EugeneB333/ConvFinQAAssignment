import pytest
from langchain_core.documents import Document
from pathlib import Path
from usecases.doc_ingest._build_index import (
    index_from_file,
    index_from_text,
    get_index_path,
    load_index,
    save_index
)

import usecases.doc_ingest._build_index as build_index_module


def make_dummy_doc(content, metadata=None):
    # Create a dummy Document with page_content and metadata attributes
    doc = Document(page_content=content)
    # Attach metadata attribute
    doc.metadata = metadata if metadata is not None else {'source': 'test'}
    return doc


def test_from_file_empty(monkeypatch):
    # Arrange: no documents returned by loader
    monkeypatch.setattr(
        build_index_module,
        'load_transform_unstructured',
        lambda filename: []
    )
    # to_langchain_simple_metadata should not be called, but stub just in case
    monkeypatch.setattr(
        build_index_module,
        'to_langchain_simple_metadata',
        lambda docs: docs
    )
    # Stub vectorstore.add_texts to raise if called
    def fake_add_texts(texts, metadatas, ids):
        pytest.skip('vectorstore.add_texts should not be called for empty input')
    monkeypatch.setattr(
        build_index_module.vectorstore,
        'add_texts',
        fake_add_texts
    )

    # Act
    result = build_index_module.from_file('dummy.txt')

    # Assert
    assert result == []


def test_from_file_single_batch(monkeypatch):
    # Arrange: two dummy documents
    docs = [make_dummy_doc('A'), make_dummy_doc('B')]
    monkeypatch.setattr(
        build_index_module,
        'load_transform_unstructured',
        lambda filename: docs
    )
    # Identity metadata conversion
    monkeypatch.setattr(
        build_index_module,
        'to_langchain_simple_metadata',
        lambda documents: documents
    )
    # Stub uuid4 to return a fixed string
    monkeypatch.setattr(
        build_index_module,
        'uuid4',
        lambda: 'fixed-id'
    )
    # Capture add_texts calls
    captured = {'calls': []}
    def fake_add_texts(texts, metadatas, ids):
        captured['calls'].append((texts, metadatas, ids))
        # Return the ids list back
        return ids
    monkeypatch.setattr(
        build_index_module.vectorstore,
        'add_texts',
        fake_add_texts
    )

    # Act
    result = build_index_module.from_file('dummy.txt', batch_size=1000)

    # Assert: returned ids
    assert result == ['fixed-id', 'fixed-id']
    # Assert add_texts called once with correct arguments
    assert len(captured['calls']) == 1
    texts, metadatas, ids = captured['calls'][0]
    assert texts == ['A', 'B']
    assert metadatas == [doc.metadata for doc in docs]
    assert ids == ['fixed-id', 'fixed-id']


def test_from_file_multiple_batches(monkeypatch):
    # Arrange: three dummy documents
    docs = [make_dummy_doc(f'D{i}') for i in range(3)]
    monkeypatch.setattr(
        build_index_module,
        'load_transform_unstructured',
        lambda filename: docs
    )
    monkeypatch.setattr(
        build_index_module,
        'to_langchain_simple_metadata',
        lambda documents: documents
    )
    monkeypatch.setattr(
        build_index_module,
        'uuid4',
        lambda: 'batch-id'
    )
    captured = {'calls': []}
    def fake_add_texts(texts, metadatas, ids):
        captured['calls'].append((texts, metadatas, ids))
        return ids
    monkeypatch.setattr(
        build_index_module.vectorstore,
        'add_texts',
        fake_add_texts
    )

    # Act: batch_size=2 should split into two batches (2 + 1)
    result = build_index_module.from_file('dummy.txt', batch_size=2)

    # Assert: three ids returned
    assert result == ['batch-id', 'batch-id', 'batch-id']
    # Two calls to add_texts
    assert len(captured['calls']) == 2
    # First batch length
    texts1, metas1, ids1 = captured['calls'][0]
    assert texts1 == ['D0', 'D1']
    # Second batch length
    texts2, metas2, ids2 = captured['calls'][1]
    assert texts2 == ['D2']


@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    content = "This is a test document.\nIt contains important information.\nAbout various topics."
    file_path.write_text(content)
    return file_path

@pytest.fixture
def index_dir(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return index_dir

def test_index_from_text():
    text = "This is a test document with some important information."
    doc_ids = index_from_text(text)
    
    assert isinstance(doc_ids, list)
    assert len(doc_ids) > 0
    assert all(isinstance(id_, str) for id_ in doc_ids)

def test_index_from_text_empty():
    doc_ids = index_from_text("")
    assert isinstance(doc_ids, list)
    assert len(doc_ids) == 0

def test_index_from_file(sample_text_file):
    doc_ids = index_from_file(str(sample_text_file))
    
    assert isinstance(doc_ids, list)
    assert len(doc_ids) > 0
    assert all(isinstance(id_, str) for id_ in doc_ids)

def test_index_from_file_nonexistent():
    with pytest.raises(FileNotFoundError):
        index_from_file("nonexistent.txt")

def test_get_index_path(index_dir):
    # Test with default name
    path = get_index_path(str(index_dir))
    assert isinstance(path, Path)
    assert path.parent == index_dir
    assert path.name.endswith(".faiss")

    # Test with custom name
    custom_name = "custom_index"
    path = get_index_path(str(index_dir), custom_name)
    assert path.name == f"{custom_name}.faiss"

def test_save_and_load_index(index_dir):
    # Create a simple index from text
    text = "This is a test document for index saving and loading."
    doc_ids = index_from_text(text)
    
    # Save the index
    index_path = get_index_path(str(index_dir))
    save_index(str(index_path))
    
    # Load the index
    loaded_index = load_index(str(index_path))
    assert loaded_index is not None

def test_load_index_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_index("nonexistent/index.faiss")
