from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, RedirectResponse

from endpoints import api_doc_ingest, api_RAG


app = FastAPI()
app.include_router(api_doc_ingest, tags=['Document Ingestion',])
app.include_router(api_RAG, tags=['Retrieval Augmented Generation'])


@app.get('/', response_class=ORJSONResponse)
async def root():
    return RedirectResponse(url='/redoc')
