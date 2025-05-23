from typing_extensions import List, Optional
from pydantic import BaseModel


class ChatDebrief(BaseModel):
    id_chat: int
    name: str
    summary: str


class ChatQuery(BaseModel):
    id_query: int
    content_query: str


class ChatResponse(BaseModel):
    id_response: int
    content_response: str


class ChatExchange(BaseModel):
    id_exchange: int
    id_chat: int
    query: ChatQuery
    response: ChatResponse


class ChatDetails(ChatDebrief):
    history: List[ChatExchange]


RelevantQueries = Optional[List[int]]
