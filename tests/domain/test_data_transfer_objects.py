import pytest

from pydantic import ValidationError

from domain._data_transfer_objects import (
    ChatDebrief,
    ChatQuery,
    ChatResponse,
    ChatExchange,
    ChatDetails,
)

from domain import (
    ChatDebriefDTO,
    ChatQueryDTO,
    ChatResponseDTO,
    ChatExchangeDTO,
    ChatDetailsDTO,
    RelevantQueriesDTO
)


def test_chatdebrief_valid():
    dto = ChatDebrief(id_chat=1, name='Test Chat', summary='A short summary')
    assert dto.id_chat == 1
    assert dto.name == 'Test Chat'
    assert dto.summary == 'A short summary'


def test_chatdebrief_invalid_missing_fields():
    with pytest.raises(ValidationError):
        ChatDebrief(name='No ID Chat', summary='Missing id_chat')


def test_chatquery_valid_and_invalid():
    # valid
    q = ChatQuery(id_query=42, content_query='Hello?')
    assert q.id_query == 42
    assert q.content_query == 'Hello?'
    # invalid: wrong type for id_query
    with pytest.raises(ValidationError):
        ChatQuery(id_query='not-an-int', content_query='Oops')


def test_chatresponse_valid_and_invalid():
    # valid
    r = ChatResponse(id_response=99, content_response='Reply')
    assert r.id_response == 99
    assert r.content_response == 'Reply'
    # invalid: missing content_response
    with pytest.raises(ValidationError):
        ChatResponse(id_response=100)


def test_chatexchange_valid_and_invalid():
    q = ChatQuery(id_query=1, content_query='Q')
    r = ChatResponse(id_response=2, content_response='R')
    ex = ChatExchange(id_exchange=10, id_chat=5, query=q, response=r)
    assert ex.id_exchange == 10
    assert ex.id_chat == 5
    assert isinstance(ex.query, ChatQuery)
    assert isinstance(ex.response, ChatResponse)
    # invalid: missing nested models
    with pytest.raises(ValidationError):
        ChatExchange(id_exchange=11, id_chat=5, query={'id_query':1}, response={'id_response':2})


def test_chatdetails_valid_and_invalid():
    q = ChatQuery(id_query=3, content_query='Hey')
    r = ChatResponse(id_response=4, content_response='Hi')
    ex = ChatExchange(id_exchange=20, id_chat=2, query=q, response=r)
    details = ChatDetails(
        id_chat=2,
        name='Detail Chat',
        summary='Detail Summary',
        history=[ex]
    )
    assert details.id_chat == 2
    assert details.name == 'Detail Chat'
    assert isinstance(details.history, list)
    assert details.history[0] == ex
    # invalid: history not a list of ChatExchange
    with pytest.raises(ValidationError):
        ChatDetails(id_chat=2, name='Bad', summary='No history', history=['not an exchange'])


def test_chat_debrief_creation():
    debrief = ChatDebriefDTO(
        id_chat=1,
        name="Test Chat",
        summary="A test chat summary"
    )
    assert debrief.id_chat == 1
    assert debrief.name == "Test Chat"
    assert debrief.summary == "A test chat summary"


def test_chat_query_creation():
    query = ChatQueryDTO(
        id_query=1,
        content_query="What is the meaning of life?"
    )
    assert query.id_query == 1
    assert query.content_query == "What is the meaning of life?"


def test_chat_response_creation():
    response = ChatResponseDTO(
        id_response=1,
        content_response="42"
    )
    assert response.id_response == 1
    assert response.content_response == "42"


def test_chat_exchange_creation():
    query = ChatQueryDTO(id_query=1, content_query="Question?")
    response = ChatResponseDTO(id_response=1, content_response="Answer!")
    exchange = ChatExchangeDTO(
        id_exchange=1,
        id_chat=1,
        query=query,
        response=response
    )
    assert exchange.id_exchange == 1
    assert exchange.id_chat == 1
    assert exchange.query == query
    assert exchange.response == response


def test_chat_details_creation():
    query = ChatQueryDTO(id_query=1, content_query="Question?")
    response = ChatResponseDTO(id_response=1, content_response="Answer!")
    exchange = ChatExchangeDTO(
        id_exchange=1,
        id_chat=1,
        query=query,
        response=response
    )
    details = ChatDetailsDTO(
        id_chat=1,
        name="Test Chat",
        summary="A test chat summary",
        history=[exchange]
    )
    assert details.id_chat == 1
    assert details.name == "Test Chat"
    assert details.summary == "A test chat summary"
    assert len(details.history) == 1
    assert details.history[0] == exchange


def test_relevant_queries_type():
    # Test None case
    queries: RelevantQueriesDTO = None
    assert queries is None

    # Test list case
    queries = [1, 2, 3]
    assert isinstance(queries, list)
    assert all(isinstance(q, int) for q in queries)


def test_chat_debrief_validation():
    with pytest.raises(ValueError):
        ChatDebriefDTO(
            id_chat="not an integer",  # type: ignore
            name="Test Chat",
            summary="A test chat summary"
        )


def test_nested_validation():
    with pytest.raises(ValueError):
        ChatExchangeDTO(
            id_exchange=1,
            id_chat=1,
            query={"id_query": "not an integer", "content_query": "Question?"},  # type: ignore
            response={"id_response": 1, "content_response": "Answer!"}
        )
