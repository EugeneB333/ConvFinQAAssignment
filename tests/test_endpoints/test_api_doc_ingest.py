import os
import base64
import pytest
from fastapi import status
from fastapi.testclient import TestClient

import endpoints._api_doc_ingest as uploads_api


def test_get_uploads_empty(client: TestClient):
    # No documents yet
    response = client.get('/uploads')
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


def test_get_upload_not_found(client: TestClient):
    # Accessing non-existent upload should return 404
    response = client.get('/uploads/nonexistent')
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()['detail'] == 'Document not found'


def test_post_upload_and_retrieval(tmp_path, client: TestClient, monkeypatch):
    # Prepare a dummy file in tmp_path
    filename = 'test.txt'
    file_content = b'hello world'
    file_path = tmp_path / filename
    file_path.write_bytes(file_content)

    # Monkeypatch the UPLOAD_DIRECTORY to tmp_path
    monkeypatch.setattr(uploads_api, 'UPLOAD_DIRECTORY', str(tmp_path))
    # Monkeypatch index_from_file to return two IDs
    dummy_ids = ['id1', 'id2']
    monkeypatch.setattr(uploads_api, 'index_from_file', lambda f: dummy_ids)

    # Perform upload via TestClient
    with open(file_path, 'rb') as f:
        files = {'file': (filename, f, 'text/plain')}
        response = client.post('/uploads', files=files)
    # Should redirect to first ID
    assert response.status_code == status.HTTP_302_FOUND
    assert response.headers['location'] == f'/uploads/{dummy_ids[0]}'

    # GET /uploads should list two entries
    response = client.get('/uploads')
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list) and len(data) == 2
    # Check first record fields
    rec0 = data[0]
    assert rec0[0][0] == 'id' and rec0[0][1] == 'id1'
    assert rec0[1][0] == 'name' and rec0[1][1] == filename
    # Contents should be base64-encoded original data
    contents_encoded = rec0[2][1]
    # Decode base64 and compare to original bytes
    decoded = base64.b64decode(contents_encoded)
    assert decoded == file_content

    # GET specific upload
    response = client.get(f'/uploads/{dummy_ids[1]}')
    assert response.status_code == status.HTTP_200_OK
    rec1 = response.json()
    assert rec1[0][1] == 'id2'
    assert rec1[1][1] == filename
    decoded1 = base64.b64decode(rec1[2][1])
    assert decoded1 == file_content


def test_upload_directory_created(tmp_path, monkeypatch):
    # Ensure that UPLOAD_DIRECTORY is created if missing
    target_dir = tmp_path / 'newuploads'
    # Remove if exists
    if target_dir.exists():
        pytest.skip('Directory unexpectedly exists')
    monkeypatch.setattr(uploads_api, 'UPLOAD_DIRECTORY', str(target_dir))
    # Reload module to trigger directory creation
    import importlib
    importlib.reload(uploads_api)
    assert target_dir.exists() and target_dir.is_dir()
