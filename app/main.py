from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.upload_json import router as upload_router
from app.api.add_json_to_chroma import router as add_json_router  # Новый импорт
from app.chroma_client.generate_chroma_db import chroma_vectorstore
from app.api.response_router import router as response_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await chroma_vectorstore.init()
    app.include_router(upload_router, prefix="/chroma_client", tags=["CROMA_CLIENT"])
    app.include_router(
        add_json_router, prefix="/chroma_db", tags=["CHROMA_DB"]
    )  # Включение нового роутера
    app.include_router(response_router, tags=["QUERY"])
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    yield
    await chroma_vectorstore.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"message": "Проект создания ИИ помощника"}
