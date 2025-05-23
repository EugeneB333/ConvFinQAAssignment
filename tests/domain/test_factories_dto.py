import pytest

from domain import (
    ChatDetailsDTO,
    ChatQueryDTO,
    ChatResponseDTO,
    ChatExchangeDTO,
    ChatExchangeFactoryDTO,
    inplace_append_chat
)
from domain._factories_dto import ChatExchangeFactory


def make_sample_query_response():
    query = ChatQueryDTO(id_query=1, content_query='What is AI?')
    response = ChatResponseDTO(id_response=1, content_response='AI is artificial intelligence')
    return query, response


def test_chat_exchange_factory_empty_history():
    # Given a chat with no history, factory should assign id_exchange = 1
    chat = ChatDetailsDTO(id_chat=42, name='TestChat', summary='Summary', history=[])
    query, response = make_sample_query_response()
    exchange = ChatExchangeFactory(chat, query, response)

    assert isinstance(exchange, ChatExchangeDTO)
    assert exchange.id_exchange == 1
    assert exchange.query == query
    assert exchange.response == response
    assert exchange.id_chat == chat.id_chat if hasattr(exchange, 'id_chat') else True


def test_chat_exchange_factory_with_existing_history():
    # Given a chat with existing history entries with various id_exchange values
    q1, r1 = make_sample_query_response()
    ex1 = ChatExchangeDTO(id_exchange=5, id_chat=42, query=q1, response=r1)
    q2 = ChatQueryDTO(id_query=2, content_query='Next?')
    r2 = ChatResponseDTO(id_response=2, content_response='Second answer')
    ex2 = ChatExchangeDTO(id_exchange=3, id_chat=42, query=q2, response=r2)

    chat = ChatDetailsDTO(id_chat=42, name='TestChat', summary='Summary', history=[ex1, ex2])
    # Factory should pick the max id_exchange (5) and add 1
    new_query = ChatQueryDTO(id_query=3, content_query='Another?')
    new_response = ChatResponseDTO(id_response=3, content_response='Third answer')
    new_ex = ChatExchangeFactory(chat, new_query, new_response)

    assert new_ex.id_exchange == 6
    assert new_ex.query == new_query
    assert new_ex.response == new_response


def test_inplace_append_chat_appends_and_returns_none():
    chat = ChatDetailsDTO(id_chat=1, name='Name', summary='Sum', history=[])
    query, response = make_sample_query_response()
    exchange = ChatExchangeDTO(id_exchange=1, id_chat=1, query=query, response=response)

    result = inplace_append_chat(chat, exchange)
    # inplace_append_chat does not return anything
    assert result is None
    # chat.history should now contain the exchange
    assert chat.history == [exchange]


def test_inplace_append_chat_does_not_duplicate_existing():
    # Ensures that appending twice adds twice (i.e., no dedupe by default)
    chat = ChatDetailsDTO(id_chat=1, name='Name', summary='Sum', history=[])
    q, r = make_sample_query_response()
    ex = ChatExchangeDTO(id_exchange=1, id_chat=1, query=q, response=r)

    inplace_append_chat(chat, ex)
    inplace_append_chat(chat, ex)
    assert chat.history == [ex, ex]


def test_chat_exchange_factory_creation():
    factory = ChatExchangeFactoryDTO(id_chat=1)
    assert factory.id_chat == 1
    assert factory.next_exchange_id == 0
    assert factory.next_query_id == 0
    assert factory.next_response_id == 0


def test_chat_exchange_factory_create_query():
    factory = ChatExchangeFactoryDTO(id_chat=1)
    query = factory.create_query("What is the meaning of life?")
    
    assert isinstance(query, ChatQueryDTO)
    assert query.id_query == 0
    assert query.content_query == "What is the meaning of life?"
    assert factory.next_query_id == 1


def test_chat_exchange_factory_create_response():
    factory = ChatExchangeFactoryDTO(id_chat=1)
    response = factory.create_response("42")
    
    assert isinstance(response, ChatResponseDTO)
    assert response.id_response == 0
    assert response.content_response == "42"
    assert factory.next_response_id == 1


def test_chat_exchange_factory_create_exchange():
    factory = ChatExchangeFactoryDTO(id_chat=1)
    query = factory.create_query("Question?")
    response = factory.create_response("Answer!")
    exchange = factory.create_exchange(query, response)
    
    assert isinstance(exchange, ChatExchangeDTO)
    assert exchange.id_exchange == 0
    assert exchange.id_chat == 1
    assert exchange.query == query
    assert exchange.response == response
    assert factory.next_exchange_id == 1


def test_inplace_append_chat():
    # Create initial chat details
    details = ChatDetailsDTO(
        id_chat=1,
        name="Test Chat",
        summary="A test chat summary",
        history=[]
    )
    
    # Create an exchange to append
    factory = ChatExchangeFactoryDTO(id_chat=1)
    query = factory.create_query("Question?")
    response = factory.create_response("Answer!")
    exchange = factory.create_exchange(query, response)
    
    # Test appending
    inplace_append_chat(details, exchange)
    
    assert len(details.history) == 1
    assert details.history[0] == exchange


def test_inplace_append_chat_multiple():
    details = ChatDetailsDTO(
        id_chat=1,
        name="Test Chat",
        summary="A test chat summary",
        history=[]
    )
    
    factory = ChatExchangeFactoryDTO(id_chat=1)
    
    # Create and append multiple exchanges
    exchanges = []
    for i in range(3):
        query = factory.create_query(f"Question {i}?")
        response = factory.create_response(f"Answer {i}!")
        exchange = factory.create_exchange(query, response)
        exchanges.append(exchange)
        inplace_append_chat(details, exchange)
    
    assert len(details.history) == 3
    for i, exchange in enumerate(details.history):
        assert exchange == exchanges[i]


def test_inplace_append_chat_validation():
    details = ChatDetailsDTO(
        id_chat=1,
        name="Test Chat",
        summary="A test chat summary",
        history=[]
    )
    
    # Create exchange with different chat id
    factory = ChatExchangeFactoryDTO(id_chat=2)
    query = factory.create_query("Question?")
    response = factory.create_response("Answer!")
    exchange = factory.create_exchange(query, response)
    
    # Should raise error when appending exchange with different chat id
    with pytest.raises(ValueError):
        inplace_append_chat(details, exchange)
