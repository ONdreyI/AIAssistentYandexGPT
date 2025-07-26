# Используем официальный образ Python в качестве базового образа
FROM python:3.11-slim-buster

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=120 --retries 5

# Копируем все файлы приложения в контейнер
COPY app/ .

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Команда для запуска приложения с помощью Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]