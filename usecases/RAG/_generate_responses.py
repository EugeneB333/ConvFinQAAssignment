from os import linesep
from uuid import uuid4
from typing import List
from langchain.memory import ConversationSummaryMemory

from infrastructure import llm_chat
from domain import ChatQueryDTO, ChatResponseDTO, HumanMessage, AIMessage, SystemMessage
from ._retrieve_context import retrieve_context


def generate_response(query: ChatQueryDTO, message_history: List[dict], debug=False) -> ChatResponseDTO:
    context_of_query = retrieve_context(query)

    messages = []

    if context_of_query:
        system_message = SystemMessage(content=f"Context: {context_of_query}")
        messages.append(system_message)

    messages.extend(message_history)

    if debug:
        class_name_to_role = {
            'HumanMessage': 'User',
            'AIMessage': 'Assistant',
            'SystemMessage': 'System',
        }
        for msg in messages:
            role = class_name_to_role.get(msg.__class__.__name__, 'Unknown')
            print(f"{role}: {msg.content}\n")

    llm_response = llm_chat.invoke(messages)

    response = ChatResponseDTO(
        id_response=uuid4().int >> 64,
        content_response=llm_response.content,
    )

    return response

