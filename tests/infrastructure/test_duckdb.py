import pytest

from infrastructure import _duckdb as db_module


def test_duckdb_connection_has_execute():
    """
    Ensure the module-level duckdb_connection exists and has an execute method.
    """
    conn = db_module.duckdb_connection
    assert hasattr(conn, 'execute'), 'duckdb_connection should have an execute() method'


def test_chats_table_schema():
    """
    Verify that the 'chats' table exists with the expected columns and types.
    """
    # Query information_schema for column definitions
    rows = db_module.duckdb_connection.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'chats'
        ORDER BY ordinal_position
        """
    ).fetchall()
    # Convert types to uppercase for comparison
    cols = [(name, dtype.upper()) for name, dtype in rows]
    expected = [
        ('id', 'BIGINT'),
        ('name', 'VARCHAR'),
        ('summary', 'VARCHAR'),
        ('history', 'VARCHAR'),
    ]
    assert cols == expected, f"Expected chats schema {expected}, got {cols}"


def test_documents_table_schema():
    """
    Verify that the 'documents' table exists with the expected columns and types.
    """
    rows = db_module.duckdb_connection.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'documents'
        ORDER BY ordinal_position
        """
    ).fetchall()
    cols = [(name, dtype.upper()) for name, dtype in rows]
    expected = [
        ('id', 'VARCHAR'),
        ('name', 'VARCHAR'),
        ('contents', 'BLOB'),
    ]
    assert cols == expected, f"Expected documents schema {expected}, got {cols}"


def test_tables_are_empty_initially():
    """
    Ensure that both tables start out empty in the fresh DB.
    """
    count_chats = db_module.duckdb_connection.execute(
        "SELECT COUNT(*) FROM chats"
    ).fetchone()[0]
    count_docs = db_module.duckdb_connection.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]
    assert count_chats == 0, 'Expected chats table to be empty initially'
    assert count_docs == 0, 'Expected documents table to be empty initially'
