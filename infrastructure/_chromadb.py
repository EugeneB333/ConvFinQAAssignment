from langchain_chroma import Chroma


PERSIST_DIRECTORY = 'db'


def get_vector_store(embeddings):
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectordb
