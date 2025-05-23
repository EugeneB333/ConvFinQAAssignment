# tests/conftest.py
import pytest
import duckdb
from fastapi import FastAPI
from fastapi.testclient import TestClient

import infrastructure as infra_pkg
from endpoints import api_doc_ingest, api_RAG

@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    """
    Every test gets a clean in-memory DuckDB with the right schema,
    and `infrastructure.localdb` is patched to point at this new conn.
    """
    conn = duckdb.connect(database=":memory:")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats ("
        "  id BIGINT PRIMARY KEY,"
        "  name VARCHAR,"
        "  summary VARCHAR,"
        "  history VARCHAR"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "  id VARCHAR PRIMARY KEY,"
        "  name VARCHAR,"
        "  contents BLOB"
        ")"
    )
    # Patch the **exported** localdb
    monkeypatch.setattr(infra_pkg, "localdb", conn)
    # Patch the module-level duckdb_connection as well
    import infrastructure._duckdb as _duckdb_mod
    monkeypatch.setattr(_duckdb_mod, "duckdb_connection", conn)
    yield conn
    conn.close()

@pytest.fixture
def app() -> FastAPI:
    """
    Construct a FastAPI() with your two routers mounted
    via their **public** names.
    """
    application = FastAPI()
    application.include_router(api_doc_ingest)
    application.include_router(api_RAG)
    return application

@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """
    A TestClient to hit your test_endpoints.
    """
    return TestClient(app)
