from domain import ChatQueryDTO, combine_langchain_docs
from infrastructure import retriever



def retrieve_context(query: ChatQueryDTO) -> str:
    retrieved_docs = retriever.invoke(query.content_query)
    formatted_context = combine_langchain_docs(retrieved_docs)
    return formatted_context
