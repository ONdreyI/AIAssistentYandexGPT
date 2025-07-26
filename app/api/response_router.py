from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from fastapi.responses import StreamingResponse

from app.api.scemas import AskResponse, AskWithAIResponse, SUserAuth
from app.chroma_client.generate_chroma_db import ChromaVectorStore, get_vectorstore
from app.chroma_client.ai_store import ChatWithAI
from app.api.utils import authenticate_user, create_jwt_token, get_current_user

router = APIRouter()


@router.post("/ask", description="Ручка для запроса")
async def ask(
    query: AskResponse,
    vectorstore: ChromaVectorStore = Depends(get_vectorstore),
    user_id: int = Depends(get_current_user),
):
    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    formatted_results = []
    for doc, score in results:
        formatted_results.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
        )
    return {"results": formatted_results}


@router.post(
    "/ask_with_ai", description="Ручка для запроса к ИИ с контекстом из Chroma"
)
async def ask_with_ai(
    query: AskWithAIResponse,
    vectorstore: ChromaVectorStore = Depends(get_vectorstore),
    user_id: int = Depends(get_current_user),
) -> StreamingResponse:
    try:
        logger.debug(
            f"Получен запрос: response='{query.response}', provider='{query.provider}', k={query.k}"
        )

        # 1. Получаем документы из Chroma
        results = await vectorstore.asimilarity_search(
            query=query.response,
            with_score=True,
            k=query.k,
        )

        # 2. Формируем AI-контекст
        if not results:
            logger.warning(
                f"Для запроса '{query.response}' не найдено документов в Chroma"
            )
            ai_context = "Контекст не найден."
        else:
            ai_context = "\n---\n".join([doc.page_content for doc, _ in results])
            logger.info(
                f"Найдено {len(results)} документов для запроса '{query.response}'"
            )

        # 3. Инициализируем ChatWithAI с выбранным провайдером
        ai_store = ChatWithAI(provider=query.provider)
        logger.info(f"Инициализирован ChatWithAI с провайдером: {query.provider}")

        # 4. Генерируем ответ от ИИ
        response = ai_store.generate_response(query.response)
        logger.debug(f"Ответ от ИИ: {response[:100]}...")
        print(f"\nИИ: {response}\n")  # Вывод в консоль для отладки

        # 5. Эмулируем StreamingResponse
        async def response_generator() -> AsyncGenerator[str, None]:
            yield response

        return StreamingResponse(
            content=response_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )

    except ValueError as ve:
        logger.error(f"Ошибка валидации в ручке /ask_with_ai: {str(ve)}")
        print(f"\nОшибка: {str(ve)}\n")

        async def error_generator() -> AsyncGenerator[str, None]:
            yield f"Ошибка валидации: {str(ve)}"

        return StreamingResponse(
            content=error_generator(),
            media_type="text/plain",
            status_code=422,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )
    except Exception as e:
        logger.error(f"Общая ошибка в ручке /ask_with_ai: {str(e)}")
        print(f"\nОшибка: {str(e)}\n")

        async def error_generator() -> AsyncGenerator[str, None]:
            yield f"Произошла ошибка: {str(e)}"

        return StreamingResponse(
            content=error_generator(),
            media_type="text/plain",
            status_code=500,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Transfer-Encoding": "chunked",
            },
        )


@router.post("/login")
async def login(
    response: Response,
    user_data: SUserAuth,
):
    user = await authenticate_user(user_data.login, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = await create_jwt_token(user["user_id"])
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,  # Включить True в проде
        samesite="lax",
        max_age=3600,
        path="/",
    )
    return {"message": "logged in"}


@router.post("/logout")
async def logout(
    response: Response,
    user_id: int = Depends(get_current_user),
):
    response.delete_cookie("access_token")
    return {"message": "Logged out"}


@router.get("/protected")
async def protected_route(user_id: int = Depends(get_current_user)):
    return {"message": f"Hello, user {user_id}!"}
