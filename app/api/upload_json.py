import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse

from config import settings

router = APIRouter()


@router.post("/upload_files")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            if not file.filename.endswith(".json"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not a JSON format"
                )

            file_pass = os.path.join(settings.PARSED_JSON_PATH, file.filename)
            with open(file_pass, "wb") as f:
                contents = await file.read()
                f.write(contents)

            uploaded_files.append(file.filename)
        return {"filenames": uploaded_files, "message": "Files uploaded successfully"}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "detail": str(e)},
        )
