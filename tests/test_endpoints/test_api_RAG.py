import json
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from domain import ChatQueryDTO, ChatResponseDTO
import endpoints._api_RAG as api


def test_get_chats_empty(client: TestClient):
    # When no chats exist, GET /chats should return empty list
    response = client.get('/chats')
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []


def test_post_new_chat_and_get_chats(client: TestClient, monkeypatch):
    # Stub generate_response to avoid external calls
    stub_response = ChatResponseDTO(id_response=123, content_response='stub')
    monkeypatch.setattr(api, 'generate_response', lambda q, h, debug: stub_response)

    # Create new chat via POST /chats/new
    response = client.post('/chats/new', json='Hello')
    assert response.status_code == status.HTTP_302_FOUND
    assert response.headers['location'] == '/chats/0'

    # After creation, GET /chats should list one chat
    response = client.get('/chats')
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list) and len(data) == 1
    chat = data[0]
    assert chat['id_chat'] == 0
    assert 'Chat #0' in chat['name']
    assert 'something wicked this way comes #0' in chat['summary']


def test_post_query_and_get_chat_and_history(client: TestClient, monkeypatch):
    # Stub generate_response to a predictable response
    stub_response = ChatResponseDTO(id_response=999, content_response='reply')
    monkeypatch.setattr(api, 'generate_response', lambda q, h, debug: stub_response)

    # Post a query to new chat id=1
    query_body = {'id_query': 0, 'content_query': 'Test?'}
    resp = client.post('/chats/1/query', json=query_body)
    assert resp.status_code == status.HTTP_302_FOUND
    assert resp.headers['location'] == '/chats/1'

    # GET /chats/1
    resp = client.get('/chats/1')
    assert resp.status_code == status.HTTP_200_OK
    chat = resp.json()
    # Should contain id_chat, name, summary, history
    assert chat['id_chat'] == 1
    assert 'Chat #1' in chat['name']
    assert 'something wicked this way comes #1' in chat['summary']
    assert isinstance(chat['history'], list) and len(chat['history']) == 1
    exch = chat['history'][0]
    # Verify exchange structure and content
    assert exch['query']['content_query'] == 'Test?'
    assert exch['response']['content_response'] == 'reply'

    # GET /chats/1/history
    resp = client.get('/chats/1/history')
    assert resp.status_code == status.HTTP_200_OK
    history = resp.json()
    assert history['id_chat'] == 1
    assert isinstance(history['history'], list) and len(history['history']) == 1


def test_get_chat_not_found(client: TestClient):
    # GET /chats/999 should return empty DTO
    resp = client.get('/chats/999')
    assert resp.status_code == status.HTTP_200_OK
    chat = resp.json()
    assert chat['id_chat'] == -1
    assert chat['name'] == ''
    assert chat['summary'] == ''
    assert chat['history'] == []

    # GET /chats/999/history should 404
    resp = client.get('/chats/999/history')
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()['detail'] == 'Chat history not found'
