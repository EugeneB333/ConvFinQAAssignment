import pytest
from functools import partial
from langchain_core.documents import Document

from usecases.doc_ingest._tokenize import LoadTransformUnstructured
import usecases.doc_ingest._tokenize as tokenize_module
from usecases.doc_ingest._tokenize import (
    tokenize_text,
    tokenize_file,
    get_supported_file_types
)


class DummyLoader:
    def __init__(self, filename, mode=None, strategy=None):
        self.filename = filename
        self.mode = mode
        self.strategy = strategy
        self.loaded = False

    def load(self):
        self.loaded = True
        # Return dummy Document objects
        return [Document(page_content=f'loaded:{self.filename}')]


class DummySplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_called = False

    def split_documents(self, docs):
        self.split_called = True
        # Append marker to each doc content
        return [Document(page_content=doc.page_content + f':split') for doc in docs]


@pytest.fixture(autouse=True)
def patch_default_loaders_and_splitters(monkeypatch):
    """
    Replace all UnstructuredFileLoader variants and text splitters with our dummies.
    """
    # Patch file loaders in tokenize_module
    monkeypatch.setattr(tokenize_module, 'UnstructuredFileLoader', DummyLoader)
    monkeypatch.setattr(tokenize_module, 'UnstructuredCSVLoader', DummyLoader)
    monkeypatch.setattr(tokenize_module, 'UnstructuredExcelLoader', DummyLoader)
    monkeypatch.setattr(tokenize_module, 'UnstructuredPDFLoader', DummyLoader)
    monkeypatch.setattr(tokenize_module, 'UnstructuredPowerPointLoader', DummyLoader)
    monkeypatch.setattr(tokenize_module, 'UnstructuredWordDocumentLoader', DummyLoader)
    # Patch text splitter classes
    monkeypatch.setattr(tokenize_module, 'RecursiveCharacterTextSplitter', DummySplitter)
    monkeypatch.setattr(tokenize_module, 'SentenceTransformersTokenTextSplitter', DummySplitter)
    yield


def test_create_loader_known_extension():
    ltu = LoadTransformUnstructured()
    # For .pdf extension, should pick UnstructuredPDFLoader
    loader_factory = ltu._create_loader('test.PDF')
    # Ensure loader_factory is a partial
    assert isinstance(loader_factory, partial)
    loader = loader_factory('test.PDF')
    assert isinstance(loader, DummyLoader)
    assert loader.filename == 'test.PDF'
    assert loader.mode == 'elements'
    assert loader.strategy == 'fast'


def test_create_loader_unknown_extension_uses_default():
    ltu = LoadTransformUnstructured()
    loader_factory = ltu._create_loader('test.unknown')
    # Should use default UnstructuredFileLoader
    assert isinstance(loader_factory, partial)
    loader = loader_factory('test.unknown')
    assert isinstance(loader, DummyLoader)
    assert loader.filename == 'test.unknown'


def test_create_splitter_default_and_specific():
    ltu = LoadTransformUnstructured()
    # Default splitter for any ext is 'characters'
    splitter_factory = ltu._create_splitter('file.txt')
    assert isinstance(splitter_factory, partial)
    splitter = splitter_factory()
    assert isinstance(splitter, DummySplitter)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 200
    # For ext mapping to characters as well
    splitter_factory2 = ltu._create_splitter('file.csv')
    splitter2 = splitter_factory2()
    assert isinstance(splitter2, DummySplitter)


def test_call_loads_and_splits_documents():
    ltu = LoadTransformUnstructured()
    # Call __call__; patch loader returns one doc, splitter appends ':split'
    result = ltu('example.pdf')
    # Should return list of Document with modified content
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].page_content == 'loaded:example.pdf:split'


def test_call_with_various_extensions_invokes_correct_loader():
    ltu = LoadTransformUnstructured()
    for ext in ['csv', 'xlsx', 'pdf', 'pptx', 'docx', 'txt']:
        filename = f'doc1.{ext}'
        result = ltu(filename)
        # Expect loader to have loaded and splitter to have split
        assert result[0].page_content.startswith(f'loaded:{filename}')
        assert result[0].page_content.endswith(':split')


def test_tokenize_text():
    text = "This is a test document. It has multiple sentences."
    tokens = tokenize_text(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)
    assert "test" in tokens
    assert "document" in tokens


def test_tokenize_text_empty():
    assert tokenize_text("") == []


def test_tokenize_text_special_chars():
    text = "Hello! This is a test-case with special@characters."
    tokens = tokenize_text(text)
    
    assert "test" in tokens
    assert "case" in tokens
    assert "special" in tokens
    assert "characters" in tokens


@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    content = "This is a test document.\nIt has multiple lines.\nAnd some content."
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_file(tmp_path):
    # Mock a PDF file - in reality you might want to use a real PDF for testing
    file_path = tmp_path / "test.pdf"
    content = b"%PDF-1.4\nSome PDF content"
    file_path.write_bytes(content)
    return file_path


def test_tokenize_file_txt(sample_text_file):
    tokens = tokenize_file(str(sample_text_file))
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "test" in tokens
    assert "document" in tokens
    assert "lines" in tokens
    assert "content" in tokens


def test_tokenize_file_unsupported():
    with pytest.raises(ValueError):
        tokenize_file("unsupported.xyz")


def test_get_supported_file_types():
    supported_types = get_supported_file_types()
    
    assert isinstance(supported_types, list)
    assert len(supported_types) > 0
    assert all(isinstance(ft, str) for ft in supported_types)
    assert ".txt" in supported_types  # At minimum, text files should be supported
