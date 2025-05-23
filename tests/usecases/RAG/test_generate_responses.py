import pytest
from uuid import UUID

from domain import ChatQueryDTO, ChatResponseDTO, HumanMessage, AIMessage, SystemMessage, ChatExchangeDTO
from infrastructure import llm_chat
from usecases.RAG._generate_responses import generate_response
from usecases.RAG._retrieve_context import retrieve_context
from usecases.RAG._generate_responses_with_summary import generate_response_with_summary

class DummyLLMResponse:
    def __init__(self, content):
        self.content = content


def test_generate_response_without_context(monkeypatch):
    # Arrange: stub retrieve_context to return empty string
    monkeypatch.setattr('usecases.RAG._generate_responses.retrieve_context', lambda q: '')
    # Prepare message history: one human message and one AI message
    history = [HumanMessage(content='Hi'), AIMessage(content='Hello')]

    # Capture invoked messages
    captured = {}
    def fake_invoke(messages):
        captured['messages'] = messages
        return DummyLLMResponse('Response content')
    monkeypatch.setattr(llm_chat, 'invoke', fake_invoke)

    # Act
    query_dto = ChatQueryDTO(id_query=1, content_query='Query?')
    response_dto = generate_response(query_dto, history, debug=False)

    # Assert: llm_chat.invoke was called with history only
    assert captured['messages'] == history
    # Assert returned DTO
    assert isinstance(response_dto, ChatResponseDTO)
    assert response_dto.content_response == 'Response content'
    assert isinstance(response_dto.id_response, int)


def test_generate_response_with_context(monkeypatch):
    # Arrange: stub retrieve_context to return some context
    monkeypatch.setattr('usecases.RAG._generate_responses.retrieve_context', lambda q: 'CTX')
    history = [HumanMessage(content='Hello1')]

    # Capture invoked messages
    captured = {}
    def fake_invoke(messages):
        captured['messages'] = messages
        return DummyLLMResponse('OK')
    monkeypatch.setattr(llm_chat_module, 'invoke', fake_invoke)

    # Act
    query_dto = ChatQueryDTO(id_query=2, content_query='What?')
    response_dto = generate_response(query_dto, history, debug=False)

    # Assert: first message is a SystemMessage with context
    assert isinstance(captured['messages'][0], SystemMessage)
    assert 'CTX' in captured['messages'][0].content
    # Followed by history messages
    assert captured['messages'][1:] == history
    # DTO correctness
    assert response_dto.content_response == 'OK'
    assert isinstance(response_dto.id_response, int)


def test_generate_response_debug_mode_prints(monkeypatch, capsys):
    # Arrange: stub retrieve_context empty, stub llm invoke
    monkeypatch.setattr('usecases.RAG._generate_responses.retrieve_context', lambda q: '')
    history = [HumanMessage(content='Msg1'), AIMessage(content='Msg2')]
    monkeypatch.setattr(llm_chat_module, 'invoke', lambda msgs: DummyLLMResponse('X'))

    # Act
    query_dto = ChatQueryDTO(id_query=3, content_query='?')
    _ = generate_response(query_dto, history, debug=True)

    # Capture stdout
    captured_out = capsys.readouterr().out
    # Should contain printed roles and content
    assert 'User: Msg1' in captured_out
    assert 'Assistant: Msg2' in captured_out


def test_generate_response_non_human_raises(monkeypatch):
    # Provide history without last element being HumanMessage
    history = [AIMessage(content='A'), AIMessage(content='B')]
    monkeypatch.setattr('usecases.RAG._generate_responses.retrieve_context', lambda q: '')
    monkeypatch.setattr(llm_chat_module, 'invoke', lambda msgs: DummyLLMResponse(''))

    with pytest.raises(ValueError):
        generate_response(ChatQueryDTO(id_query=4, content_query='err'), history, debug=False)


def test_generate_response():
    query = ChatQueryDTO(id_query=1, content_query="What is the meaning of life?")
    history = []  # Empty history for first query
    
    response = generate_response(query, history)
    
    assert isinstance(response, ChatResponseDTO)
    assert response.id_response >= 0
    assert isinstance(response.content_response, str)
    assert len(response.content_response) > 0


def test_generate_response_with_history():
    # Create a history of exchanges
    query1 = ChatQueryDTO(id_query=1, content_query="What is Python?")
    response1 = ChatResponseDTO(id_response=1, content_response="Python is a programming language.")
    exchange1 = ChatExchangeDTO(id_exchange=1, id_chat=1, query=query1, response=response1)
    
    # New query
    query2 = ChatQueryDTO(id_query=2, content_query="What are its main features?")
    
    response2 = generate_response(query2, [exchange1])
    
    assert isinstance(response2, ChatResponseDTO)
    assert response2.id_response >= 0
    assert isinstance(response2.content_response, str)
    assert len(response2.content_response) > 0


def test_generate_response_with_summary():
    query = ChatQueryDTO(id_query=1, content_query="What is the meaning of life?")
    history = []  # Empty history for first query
    
    response = generate_response_with_summary(query, history)
    
    assert isinstance(response, ChatResponseDTO)
    assert response.id_response >= 0
    assert isinstance(response.content_response, str)
    assert len(response.content_response) > 0


def test_generate_response_with_summary_and_history():
    # Create a history of exchanges
    query1 = ChatQueryDTO(id_query=1, content_query="What is Python?")
    response1 = ChatResponseDTO(id_response=1, content_response="Python is a programming language.")
    exchange1 = ChatExchangeDTO(id_exchange=1, id_chat=1, query=query1, response=response1)
    
    # New query
    query2 = ChatQueryDTO(id_query=2, content_query="What are its main features?")
    
    response2 = generate_response_with_summary(query2, [exchange1])
    
    assert isinstance(response2, ChatResponseDTO)
    assert response2.id_response >= 0
    assert isinstance(response2.content_response, str)
    assert len(response2.content_response) > 0


def test_generate_response_empty_query():
    query = ChatQueryDTO(id_query=1, content_query="")
    
    with pytest.raises(ValueError):
        generate_response(query, [])
    
    with pytest.raises(ValueError):
        generate_response_with_summary(query, [])
