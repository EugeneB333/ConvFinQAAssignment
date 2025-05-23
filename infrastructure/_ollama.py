from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama


ollama_chat = ChatOllama(base_url='http://172.235.38.66:8080', model='llama3-chatqa')
ollama_embeddings = OllamaEmbeddings(model='llama3-chatqa', base_url='http://172.235.38.66:8080', show_progress=True)
