import json
from uuid import uuid4
from typing import List, Dict, Any

from tqdm import tqdm
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from infrastructure import vectorstore
from domain import to_langchain_simple_metadata
from ._tokenize import LoadTransformUnstructured


load_transform_unstructured = LoadTransformUnstructured()


def _transform_json_entries(entry: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []

    pre = "\n".join(entry.get("pre_text", []))
    post = "\n".join(entry.get("post_text", []))
    table_lines = [" | ".join(map(str, row)) for row in entry.get("table", [])]
    table_str = "\n".join(table_lines)

    qa = entry.get("qa", {})
    qa_str = ""
    if qa.get("question") and qa.get("answer"):
        qa_str = f"\nQ: {qa['question']}\nA: {qa['answer']}"

    content = "\n\n".join(filter(None, [pre, post, "TABLE:", table_str, qa_str]))
    metadata = {
        "id": entry.get("id"),
        "source_file": entry.get("filename"),
    }
    docs.append(Document(page_content=content, metadata=metadata))
    return docs


def from_file(
    filename: str,
    batch_size: int = 1000,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[str]:

    lower = filename.lower()
    is_json = lower.endswith(".json") or lower.endswith(".jsonl")

    if is_json:
        with open(filename, "r", encoding="utf-8") as f:
            entries = json.load(f)
        documents = _transform_json_entries(entries)
    else:
        documents = load_transform_unstructured(filename=filename)

    documents = to_langchain_simple_metadata(documents=documents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.split_documents(documents)

    ids_added: List[str] = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    if is_json:
        for i in range(0, len(documents)):
            batch = documents[i : i + batch_size]
            batch_ids = vectorstore.add_texts(
                texts=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
                ids=[str(uuid4()) for _ in batch],
            )
            ids_added.extend(batch_ids)
    else:
        for i in tqdm(range(0, len(documents), batch_size),
                      desc="Ingestion batches",
                      total=total_batches,
                      unit="batch"):
            batch = documents[i: i + batch_size]
            batch_ids = vectorstore.add_texts(
                texts=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
                ids=[str(uuid4()) for _ in batch],
            )
            ids_added.extend(batch_ids)

    return ids_added
