import os
from typing import List
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
openai_api_key=os.getenv('OPENAI_API_KEY')


class MessageAwareRAG:
    def __init__(self, retriever: VectorStoreRetriever, openai_api_key: str, model_name: str = "gpt-4", temperature: float = 0.0):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
        self.retriever = retriever

    def invoke(self, messages: List[BaseMessage]):
        history = messages[:-1]
        query_msg = messages[-1] # the current HumanMessage

        if not isinstance(query_msg, HumanMessage):
            raise ValueError(
                "Last message must be a HumanMessage representing the current user query.")

        history_str = ""
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_str += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"

        docs = self.retriever.get_relevant_documents(query_msg.content)
        context = "\n".join(doc.page_content for doc in docs)

        system_prompt = SystemMessage(content=f"""You are a helpful assistant. Use the conversation history and the retrieved context to answer the user's question.

    Conversation History:
    {history_str}

    Context:
    {context}
    """)

        final_messages = [
            system_prompt,
            query_msg
        ]

        return self.llm.invoke(final_messages)


# Your new get_llm_chain
def get_llm_chain(retriever):
    return MessageAwareRAG(retriever=retriever, openai_api_key=openai_api_key)
