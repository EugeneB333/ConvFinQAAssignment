import os
import json
import shutil

from fastapi import APIRouter, Path, status, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse, RedirectResponse
from typing_extensions import Annotated
from typing import Any, List, Tuple
from tqdm import tqdm

from usecases.doc_ingest import index_from_file
from infrastructure import localdb


router = APIRouter()

UPLOAD_DIRECTORY = 'uploads'
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@router.get('/uploads', response_class=ORJSONResponse)
async def get_uploads() -> List[List[Tuple[str, Any]]]:
    # Load all documents into a DataFrame for named access
    df = localdb.execute(
        "SELECT id, name, contents FROM documents ORDER BY id"
    ).df()
    return [
        [
            ('id', rec['id']),
            ('name', rec['name']),
            ('contents', rec['contents']),
        ]
        for rec in df.to_dict(orient='records')
    ]

@router.get('/uploads/{_id}', response_class=ORJSONResponse)
async def get_upload(_id: Annotated[str, Path]) -> List[Tuple[str, Any]]:
    df = localdb.execute(
        "SELECT id, name, contents FROM documents WHERE id = ?", [_id]
    ).df()
    if df.empty:
        raise HTTPException(status_code=404, detail='Document not found')

    rec = df.to_dict(orient='records')[0]
    return [
        ('id', rec['id']),
        ('name', rec['name']),
        ('contents', rec['contents']),
    ]

@router.post('/uploads', response_class=ORJSONResponse)
async def post_upload(file: UploadFile) -> RedirectResponse:
    dest = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(dest, 'wb') as buf:
        shutil.copyfileobj(file.file, buf)

    lower = dest.lower()
    is_json = lower.endswith(".json") or lower.endswith(".jsonl")

    if is_json:
        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, entry in enumerate(tqdm(data, desc="Processing JSON", unit="entry"), start=1):
            filename = f"{dest.split('.')[0]}_{idx}.json"
            filepath = filename
            with open(filepath, 'w', encoding='utf-8') as out_f:
                json.dump(entry, out_f, ensure_ascii=False, indent=2)

            ids_indexed = index_from_file(filepath)

            if not ids_indexed:
                raise HTTPException(
                    status_code=400,
                    detail="Upload succeeded but no documents were ingested"
                )

            with open(dest, 'rb') as f:
                data = f.read()

            for _id in ids_indexed:
                exists_df = localdb.execute(
                    "SELECT 1 FROM documents WHERE id = ?", [_id]
                ).df()
                if not exists_df.empty:
                    continue

                localdb.execute(
                    "INSERT INTO documents (id, name, contents) VALUES (?, ?, ?)",
                    [_id, file.filename, data],
                )
    else:
        ids_indexed = index_from_file(dest)

        if not ids_indexed:
            raise HTTPException(
                status_code=400,
                detail="Upload succeeded but no documents were ingested"
            )

        with open(dest, 'rb') as f:
            data = f.read()

        for _id in tqdm(ids_indexed, desc="Uploading documents", unit="doc"):
            exists_df = localdb.execute(
                "SELECT 1 FROM documents WHERE id = ?", [_id]
            ).df()
            if not exists_df.empty:
                continue

            localdb.execute(
                "INSERT INTO documents (id, name, contents) VALUES (?, ?, ?)",
                [_id, file.filename, data],
            )

    print("Upload complete")

    return RedirectResponse(
        url=f'/uploads/{ids_indexed[0]}', status_code=status.HTTP_302_FOUND
    )
