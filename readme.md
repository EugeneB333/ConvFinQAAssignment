# RAG JSON Table Agent

**RAG JSON Table Agent** is a Retrieval-Augmented Generation (RAG) system designed to answer questions by retrieving and reasoning over tabular data stored in JSON files. It is built for modularity, extensibility, and clarity, making it easy to adapt for new data sources, LLMs, or workflows. The project is API-first, with a clean separation between domain logic, infrastructure, use cases, and endpoints.

**Key Use Cases:**
- Semantic Q&A over structured tabular data (e.g., financial, scientific, or business tables in JSON)
- Rapid prototyping of RAG pipelines
- Educational or research projects on LLM-based retrieval

---

## Architecture Diagram

```
+----------------+      +----------------+      +----------------+      +----------------+      +----------------+
|   User/API     | ---> |  Ingestion     | ---> |  Vector Store  | ---> |   Retrieval    | ---> |     LLM        |
| (HTTP Client)  |      | (JSON Upload)  |      |  (ChromaDB)    |      | (Semantic      |      | (OpenAI, etc.) |
|                |      |                |      |                |      |  Search)       |      |                |
+----------------+      +----------------+      +----------------+      +----------------+      +----------------+
```

---

## Project Structure

```bash
├── domain
│   ├── _data_transfer_objects.py       # DTO definitions
│   ├── _factories_dto.py               # DTO factory functions
│   ├── _langchain.py                   # Domain workflows using LangChain
│   └── __init__.py
├── infrastructure
│   ├── _chromadb.py                    # ChromaDB vector store setup
│   ├── _duckdb.py                      # duckDB SQL engine integration
│   ├── _openai.py                      # OpenAI client wrapper
│   └── __init__.py
├── usecases
│   ├── doc_ingest
│   │   ├── _build_index.py             # Build embeddings index from JSON
│   │   ├── _tokenize.py                # Tokenization logic for tables
│   │   └── __init__.py
│   └── RAG
│       ├── _retrieve_context.py        # Retrieve relevant context rows
│       ├── _generate_responses.py      # Generate answers via LLM
│       └── __init__.py
├── endpoints
│   ├── _api_doc_ingest.py              # /ingest JSON endpoint
│   ├── _api_RAG.py                     # /ask question endpoint
│   └── __init__.py
├── tests                              # Unit and integration tests
│   ├── domain
│   ├── infrastructure
│   ├── usecases
│   └── test_endpoints
├── main.py                            # FastAPI application entrypoint
├── requirements.txt
├── pyproject.toml
└── readme.md                          # This README
```

Each folder encapsulates a layer of the architecture, with corresponding tests located under `tests/` to ensure full coverage.

---

## API Documentation

### Document Ingestion
- **POST `/uploads`**: Upload a JSON or supported file. Triggers ingestion and indexing.
  - **Request**: Multipart file upload.
  - **Response**: Redirects to `/uploads/{id}` for the first ingested document.
- **GET `/uploads`**: List all uploaded documents.
- **GET `/uploads/{id}`**: Get details of a specific uploaded document.

### Chat & Retrieval
- **POST `/chats/new`**: Start a new chat with an initial query.
  - **Request**: `{ "query": "<your question>" }`
  - **Response**: Redirects to `/chats/{id}`.
- **POST `/chats/{id}/query`**: Ask a follow-up question in an existing chat.
  - **Request**: `{ "content_query": "<your question>" }`
- **GET `/chats`**: List all chats (id, name, summary).
- **GET `/chats/{id}`**: Get chat details and history.
- **GET `/chats/{id}/history`**: Get only the chat history.

**Interactive API docs:**
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Data Flow Example

1. **Upload a JSON Table**
   - Use `/uploads` to upload your file (e.g., via Swagger UI or `curl`).
2. **Ingestion**
   - The file is parsed, split, and indexed into ChromaDB (vector store) and duckDB (SQL store).
3. **Ask a Question**
   - Start a chat with `/chats/new` or continue with `/chats/{id}/query`.
   - The system retrieves relevant rows, constructs context, and sends it to the LLM.
4. **Get a Response**
   - The LLM generates an answer, which is returned and stored in chat history.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EugeneB333/ConvFinQAAssignment.git
   cd ConvFinQAAssignment

2. **Create and activate a virtual environment**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt

## Configuration & Environment

- **Environment Variables:**
  - `OPENAI_API_KEY`: Required for OpenAI LLM access. Set in your shell or `.env` file.
- **Folders:**
  - `uploads/`: Must exist and be writable for file uploads; created automatically.
  - `db/`: Used for persistent vector and SQL storage; created automatically.
- **Dependencies:**
  - See `requirements.txt` for all required packages.
  - Main: `fastapi`, `hypercorn`, `langchain`, `chromadb`, `duckdb`, `openai`, `tqdm`, `python-multipart`, `ijson`, etc.

---

## Troubleshooting & FAQ

- **Missing API Key:**
  - Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.
- **Vector DB errors:**
  - Check that `db/` exists and is writable. Ensure ChromaDB is installed.
- **File upload issues:**
  - Confirm `uploads/` exists and has correct permissions.
- **LLM errors or timeouts:**
  - Check your OpenAI usage limits and network connection.
- **General debugging:**
  - Use FastAPI's interactive docs for endpoint testing and error messages.

---

## Dependencies

- **FastAPI**: Web framework powering the HTTP API with automatic OpenAPI docs. [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- **ChromaDB**: In-memory vector database for fast embedding-based retrieval.  [ChromaDB Documentation](https://docs.trychroma.com/)
- **duckDB**: Embedded SQL database optimized for analytics on JSON-structured tables. [duckDB Documentation](https://duckdb.org/docs/)
- **OpenAI**: Large language models for query interpretation and answer generation. [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Usage 

### Start the FastAPI server

    ```bash
    hypercorn main:app --reload --worker-class trio```

Navigate to the interactive docs

    http://localhost:8000/docs

## Future Directions

- **SQL-Generating Agent**: Automatically generate and execute SQL queries based on user intent for precise table operations.  

## Notes
### OpenAI

- **Low barrier to entry**  
  Start experimenting immediately via a simple HTTP API -- no need to provision, configure, or maintain GPUs or servers as you would for self-hosted models like LLaMA.

- **Pay-as-you-go pricing**  
  Only pay for the tokens you consume. You can prototype for just a few cents instead of investing in hardware or cloud VMs to run LLaMA from day one.

- **Managed scaling & reliability**  
  Automatic load-balancing, model updates, and uptime SLAs handled by OpenAI -- so you focus on your product, not on the operational overhead of hosting LLaMA.

- **Access to cutting-edge models**  
  Instantly tap into the latest improvements (GPT-4, embeddings, fine-tuning) without manually downloading, converting, or integrating large LLaMA checkpoint files.

- **Built-in safety & compliance**  
  Leverage OpenAI’s moderation, data-handling policies, and certifications rather than building your own safeguards around a self-hosted LLaMA instance.
