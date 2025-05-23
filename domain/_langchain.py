from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from typing_extensions import List
from os import linesep


def combine_docs(docs: List[Document]) -> str:
    return f"{linesep}{linesep}".join(doc.page_content for doc in docs)
