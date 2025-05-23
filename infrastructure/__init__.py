from ._openai import get_llm_chain, OpenAIEmbeddings
from ._duckdb import duckdb_connection as localdb
from ._chromadb import get_vector_store


embeddings = OpenAIEmbeddings()
vectorstore = get_vector_store(embeddings)
retriever = vectorstore.as_retriever()

llm_chat = get_llm_chain(retriever=retriever)

__all__ = ('localdb', 'llm_chat', 'retriever', 'vectorstore',)
