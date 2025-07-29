from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.upload_json import router as upload_router
from app.api.add_json_to_chroma import router as add_json_router
from app.chroma_client.generate_chroma_db import chroma_vectorstore
from app.api.response_router import router as response_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await chroma_vectorstore.init()
    app.include_router(upload_router, prefix="/chroma_client", tags=["CROMA_CLIENT"])
    app.include_router(add_json_router, prefix="/chroma_db", tags=["CHROMA_DB"])
    app.include_router(response_router, prefix="/api", tags=["QUERY"])
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    yield
    await chroma_vectorstore.close()


app = FastAPI(lifespan=lifespan)


@app.on_event("startup")
async def startup_event():
    app.mount("/static", StaticFiles(directory="app/static"), name="static")


templates = Jinja2Templates(directory="app/templates")


# Маршрут для главной страницы
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.get("/") # Удалите или закомментируйте этот дублирующийся маршрут
# async def read_root():
#     return {"message": "Проект создания ИИ помощника"}


# Маршрут для страницы логина
@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
