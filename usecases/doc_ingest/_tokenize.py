from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
)
from langchain_community.document_loaders import (
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from typing_extensions import List, Type
from collections import defaultdict
from functools import partial


class LoadTransformUnstructured:

    def __call__(self, filename: str) -> List[Document]:
        document_loader: Type[UnstructuredFileLoader] = self._create_loader(filename)
        text_splitter: Type[TextSplitter] = self._create_splitter(filename)
        documents: List[Document] = document_loader(filename).load()
        documents = text_splitter().split_documents(documents)
        return documents

    def __init__(self):
        self.__document_loaders = defaultdict(lambda: UnstructuredFileLoader)
        self.__document_loaders.update({
            'csv': UnstructuredCSVLoader,
            'xlsx': UnstructuredExcelLoader,
            'pdf': UnstructuredPDFLoader,
            'pptx': UnstructuredPowerPointLoader,
            'docx': UnstructuredWordDocumentLoader,
        })
        self.document_splitters = defaultdict(lambda: 'characters')
        self.document_splitters.update({
            'csv': 'characters',
            'xlsx': 'characters',
            'pdf': 'characters',
            'pptx': 'characters',
            'docx': 'characters',
        })
        self.__text_splitters = defaultdict(lambda: TextSplitter)
        self.__text_splitters.update({
            'characters': RecursiveCharacterTextSplitter,
            'sentence': SentenceTransformersTokenTextSplitter,
        })

    def _create_loader(self, filename: str) -> Type[UnstructuredFileLoader]:
        file_extension = filename.split('.')[-1].lower()
        document_loader = self.__document_loaders[file_extension]
        document_loader = partial(document_loader, mode='elements', strategy='fast')
        return document_loader

    def _create_splitter(self, filename: str) -> Type[TextSplitter]:
        file_extension = filename.split('.')[-1].lower()
        document_splitter = self.document_splitters[file_extension]
        text_splitter = self.__text_splitters[document_splitter]
        text_splitter = partial(text_splitter, chunk_size=500, chunk_overlap=200)
        return text_splitter
