import json
import pandas as pd
from uuid import uuid4
from typing_extensions import Annotated, List
from fastapi import APIRouter, Body, Path, status, HTTPException
from fastapi.responses import ORJSONResponse, RedirectResponse
from langchain_community.chat_message_histories import ChatMessageHistory

from domain import (
    ChatDebriefDTO,
    ChatDetailsDTO,
    ChatQueryDTO,
    ChatResponseDTO,
    ChatExchangeDTO,
)
from infrastructure import localdb
from usecases.RAG import generate_response


router = APIRouter()

@router.get('/chats', response_class=ORJSONResponse)
async def get_chats() -> List[ChatDebriefDTO]:
    # Load into DataFrame for named access
    df = localdb.execute(
        "SELECT id, name, summary FROM chats ORDER BY id"
    ).df()
    # Convert each row to ChatDebriefDTO by column name
    return [
        ChatDebriefDTO(
            id_chat=int(rec['id']),
            name=rec['name'],
            summary=rec['summary'],
        )
        for rec in df.to_dict(orient='records')
    ]

@router.get('/chats/{_id}', response_class=ORJSONResponse)
async def get_chat(_id: Annotated[int, Path]) -> ChatDetailsDTO:
    df = localdb.execute(
        "SELECT id, name, summary, history FROM chats WHERE id = ?", [_id]
    ).df()
    if df.empty:
        return ChatDetailsDTO(
            id_chat=-1, name='', summary='', history=[]
        )

    record = df.to_dict(orient='records')[0]
    history_items = json.loads(record.get('history') or '[]')
    history = [ChatExchangeDTO.parse_obj(item) for item in history_items]
    return ChatDetailsDTO(
        id_chat=int(record['id']),
        name=record['name'],
        summary=record['summary'],
        history=history,
    )

@router.post('/chats/{_id}/query', response_class=ORJSONResponse)
async def post_query(
    _id: Annotated[int, Path],
    query_chat: Annotated[ChatQueryDTO, Body],
) -> RedirectResponse:
    df = localdb.execute(
        "SELECT history, summary FROM chats WHERE id = ?", [_id]
    ).df()
    if not df.empty:
        rec = df.to_dict(orient='records')[0]
        history_list = json.loads(rec.get('history') or '[]')
        counter = int(rec.get('summary', '#0').split('#')[-1])
    else:
        history_list = []
        counter = 0

    chat_history = ChatMessageHistory()
    for item in history_list:
        ex = ChatExchangeDTO.parse_obj(item)
        chat_history.add_user_message(ex.query.content_query)
        chat_history.add_ai_message(ex.response.content_response)

    chat_history.add_user_message(query_chat.content_query)
    llm_response = generate_response(query_chat, chat_history.messages, debug=True)

    exchange = ChatExchangeDTO(
        id_chat=_id,
        id_exchange=uuid4().int >> 64,
        query=query_chat,
        response=llm_response,
    )
    history_list.append(exchange.dict())
    new_summary = f'something wicked this way comes #{counter + 1}'
    new_name = f'Chat #{_id} (id_chat: {_id})'
    serialized_history = json.dumps(history_list)

    if not df.empty:
        localdb.execute(
            "UPDATE chats SET name = ?, summary = ?, history = ? WHERE id = ?",
            [new_name, new_summary, serialized_history, _id],
        )
    else:
        localdb.execute(
            "INSERT INTO chats (id, name, summary, history) VALUES (?, ?, ?, ?)",
            [_id, new_name, new_summary, serialized_history],
        )

    return RedirectResponse(url=f'/chats/{_id}', status_code=status.HTTP_302_FOUND)

@router.post('/chats/new', response_class=ORJSONResponse)
async def post_new_chat(query: Annotated[str, Body]) -> RedirectResponse:
    df = localdb.execute("SELECT MAX(id) AS max_id FROM chats").df()
    max_id = df.at[0, 'max_id'] if not df.empty and pd.notna(df.at[0, 'max_id']) else None
    next_id = int(max_id) + 1 if max_id is not None else 0
    return await post_query(_id=next_id, query_chat=ChatQueryDTO(id_query=0, content_query=query))

@router.get('/chats/{_id}/history', response_class=ORJSONResponse)
async def get_chat_history(_id: Annotated[int, Path]) -> ChatDetailsDTO:
    df = localdb.execute(
        "SELECT id, name, summary, history FROM chats WHERE id = ?", [_id]
    ).df()
    if df.empty:
        raise HTTPException(status_code=404, detail='Chat history not found')

    rec = df.to_dict(orient='records')[0]
    history_items = json.loads(rec.get('history') or '[]')
    history = [ChatExchangeDTO.parse_obj(item) for item in history_items]
    return ChatDetailsDTO(
        id_chat=int(rec['id']), name=rec['name'], summary=rec['summary'], history=history
    )
