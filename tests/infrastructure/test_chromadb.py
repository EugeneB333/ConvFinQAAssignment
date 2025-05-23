import pytest

import infrastructure._chromadb as chroma_module


def test_persist_directory_constant():
    # Ensure the PERSIST_DIRECTORY constant exists and is a non-empty string
    assert hasattr(chroma_module, 'PERSIST_DIRECTORY')
    persist_dir = chroma_module.PERSIST_DIRECTORY
    assert isinstance(persist_dir, str)
    assert persist_dir != ''


def test_get_vector_store_invokes_Chroma_with_correct_args(monkeypatch):
    # Capture the arguments passed to Chroma
    captured = {}

    class DummyChroma:
        def __init__(self, persist_directory, embedding_function):
            captured['persist_directory'] = persist_directory
            captured['embedding_function'] = embedding_function

    # Monkeypatch the Chroma class in the module
    monkeypatch.setattr(chroma_module, 'Chroma', DummyChroma)

    # Use a sentinel for embeddings
    fake_embeddings = object()

    vs = chroma_module.get_vector_store(fake_embeddings)

    # The returned object should be an instance of our dummy
    assert isinstance(vs, DummyChroma)
    # Check that it was called with the module's PERSIST_DIRECTORY
    assert captured.get('persist_directory') == chroma_module.PERSIST_DIRECTORY
    # Check that the embedding function argument was passed through
    assert captured.get('embedding_function') is fake_embeddings


def test_get_vector_store_returns_Chroma_instance(monkeypatch):
    # Verify that get_vector_store returns whatever Chroma returns
    class DummyChroma2:
        pass

    monkeypatch.setattr(chroma_module, 'Chroma', lambda **kwargs: DummyChroma2())

    vs = chroma_module.get_vector_store(embeddings=None)
    assert isinstance(vs, DummyChroma2)
