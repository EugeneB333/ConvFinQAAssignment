import pytest
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import infrastructure._openai as openai_module


class DummyLLM:
    def __init__(self, openai_api_key: str, model_name: str, temperature: float):
        # capture constructor args
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.invoked_with: List = []

    def invoke(self, messages: List):
        # capture the messages and return a dummy response
        self.invoked_with = messages
        return 'dummy_response'


class DummyRetriever:
    def __init__(self, docs_contents):
        # docs_contents: list of strings
        self.docs_contents = docs_contents

    def get_relevant_documents(self, query: str):
        # ignore query, return dummy docs with page_content attr
        class Doc:
            def __init__(self, content):
                self.page_content = content

        return [Doc(c) for c in self.docs_contents]


@pytest.fixture(autouse=True)
def patch_chat_openai(monkeypatch):
    # Monkeypatch ChatOpenAI in the module to our DummyLLM
    monkeypatch.setattr(openai_module, 'ChatOpenAI', DummyLLM)
    yield


def test_get_llm_chain_returns_rag_instance():
    retriever = DummyRetriever([])
    rag = openai_module.get_llm_chain(retriever)
    assert isinstance(rag, openai_module.MessageAwareRAG)
    assert rag.retriever is retriever
    # DummyLLM constructed with api key from env; ensure llm is DummyLLM
    assert isinstance(rag.llm, DummyLLM)
    assert rag.llm.openai_api_key == openai_module.openai_api_key


def test_invoke_raises_on_non_human_last():
    retriever = DummyRetriever([])
    rag = openai_module.MessageAwareRAG(retriever, openai_api_key='key')
    messages = [SystemMessage(content='sys'), AIMessage(content='ai')]
    with pytest.raises(ValueError):
        rag.invoke(messages)


def test_invoke_calls_llm_with_correct_messages_and_context():
    # Prepare history and query
    history = [HumanMessage(content='Hi'), AIMessage(content='Hello back')]
    query = HumanMessage(content='What is 2+2?')
    # Retriever returns context docs
    docs = ['Doc1', 'Doc2']
    retriever = DummyRetriever(docs)
    rag = openai_module.MessageAwareRAG(retriever, openai_api_key='k')

    # Invoke
    result = rag.invoke(history + [query])

    # Should return dummy_response
    assert result == 'dummy_response'

    # Check that llm.invoke was called with [SystemMessage, query]
    invoked = rag.llm.invoked_with
    assert len(invoked) == 2
    sys_msg, query_msg = invoked
    assert isinstance(sys_msg, SystemMessage)
    assert query_msg is query

    # Verify system prompt content includes history and context
    sys_text = sys_msg.content
    # History formatting: 'User: Hi', 'Assistant: Hello back'
    assert 'User: Hi' in sys_text
    assert 'Assistant: Hello back' in sys_text
    # Context formatting: joined docs
    assert 'Doc1' in sys_text
    assert 'Doc2' in sys_text
