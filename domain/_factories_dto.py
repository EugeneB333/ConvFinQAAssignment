from domain import ChatDetailsDTO, ChatQueryDTO, ChatResponseDTO, ChatExchangeDTO

# TODO: Need a factory for each of chat response and each query
# TODO: IDs only need to be unique, don't need to be incremented

class ChatExchangeFactory(object):
    def __new__(cls, chat: ChatDetailsDTO, query: ChatQueryDTO, response: ChatResponseDTO):
        if len(chat.history):
            id_exchange_latest: int = sorted(chat.history, key=lambda ce: ce.id_exchange, reverse=True)[0].id_exchange
        else:
            id_exchange_latest: int = 0
        return ChatExchangeDTO(id_exchange=id_exchange_latest+1, query=query, response=response)


def inplace_append_chat(chat: ChatDetailsDTO, chat_exchange: ChatExchangeDTO):
    chat.history += [chat_exchange, ]
