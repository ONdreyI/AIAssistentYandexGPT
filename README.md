# LLM_Yandex Project

This project is a FastAPI application designed to interact with YandexGPT, manage ChromaDB for document storage and retrieval, and provide a web interface for chat and document processing.

## Описание проекта (Russian)

LLM_Yandex — это веб-приложение на базе FastAPI, разработанное для взаимодействия с YandexGPT и другими моделями искусственного интеллекта. Проект представляет собой интеллектуального ассистента, который может отвечать на вопросы пользователей, используя контекст из загруженных документов.

### Основные возможности

- **Обработка и хранение документов** — система позволяет загружать JSON-документы, обрабатывать их и сохранять в векторной базе данных ChromaDB для последующего семантического поиска.
- **Семантический поиск** — поиск релевантной информации в документах на основе векторных эмбеддингов с использованием модели sentence-transformers.
- **Генерация ответов с помощью ИИ** — интеграция с различными моделями ИИ (YandexGPT, DeepSeek, OpenAI, Llama) для генерации ответов на основе найденного контекста.
- **Аутентификация пользователей** — система включает механизм аутентификации на основе JWT-токенов.
- **Веб-интерфейс** — предоставление удобного интерфейса для взаимодействия с системой через браузер.

### Технологии

- **Бэкенд**: FastAPI, Pydantic, JWT, Jinja2
- **Работа с ИИ и данными**: LangChain, ChromaDB, HuggingFace Embeddings, YandexGPT API
- **Инфраструктура**: Docker, Docker Compose, Git
- **Фронтенд**: HTML/CSS/JavaScript

## Project Structure

- `app/`: Contains the main application logic.
  - `api/`: API endpoints for interacting with the application (e.g., adding JSON to ChromaDB, handling responses).
  - `chroma_client/`: Logic for interacting with ChromaDB and YandexGPT.
  - `config.py`: Application configuration settings.
  - `main.py`: Main FastAPI application entry point.
  - `parsed_json/`: Directory for parsed JSON documents (intended to be empty in the Docker image).
  - `static/`: Static files (CSS, JavaScript).
  - `templates/`: HTML templates.
- `chroma_db/`: Directory for ChromaDB data (intended to be empty in the Docker image).
- `Dockerfile`: Dockerfile for building the application's Docker image.
- `requirements.txt`: Python dependencies.
- `.env`: Environment variables (should not be committed to version control).
- `.gitignore`: Specifies intentionally untracked files to ignore by Git.
- `.dockerignore`: Specifies files and directories to exclude when building Docker images.

## Setup (Local Development)

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd LLM_Yandex
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    # source .venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    Create a file named `.env` in the `app/` directory and add your YandexGPT API key and folder ID:
    ```
    YANDEX_GPT_API_KEY=your_api_key_here
    YANDEX_GPT_FOLDER_ID=your_folder_id_here
    ```

5.  **Run the application:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The application will be accessible at `http://127.0.0.1:8000`.

## Docker

To build and run the application using Docker:

1.  **Build the Docker image:**
    Make sure you are in the root directory of the project (`c:\Users\psk-a\LLM_Yandex`).
    ```bash
    docker build -t AI_agent_yandexGPT .
    ```

2.  **Run the Docker container:**
    Remember to provide your YandexGPT API key and folder ID as environment variables.
    ```bash
    docker run -p 8000:8000 -e YANDEX_GPT_API_KEY=your_api_key -e YANDEX_GPT_FOLDER_ID=your_folder_id AI_agent_yandexGPT
    ```
    The application will be accessible at `http://localhost:8000`.

## Usage

-   Access the web interface at `/`.
-   Use the API endpoints (e.g., `/api/upload_json`, `/api/chat`) for programmatic interaction.

## Contributing

Feel free to contribute to this project. Please follow standard Git practices for pull requests and issue reporting.

## License

[Specify your license here, e.g., MIT License]