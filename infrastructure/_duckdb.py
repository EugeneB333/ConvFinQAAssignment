import duckdb


duckdb_connection = duckdb.connect(':memory:', read_only=False)
duckdb_connection.execute(
    '''
    CREATE TABLE IF NOT EXISTS chats (
        id BIGINT PRIMARY KEY,
        name VARCHAR,
        summary VARCHAR,
        history VARCHAR
    )
    '''
)
duckdb_connection.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    contents BLOB
)
''')
