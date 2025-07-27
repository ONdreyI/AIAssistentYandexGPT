# Используем базовый образ с Debian Bullseye
FROM python:3.11-bullseye

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем torch отдельно, затем остальные зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=120 --retries 5 \
    pip cache purge

    


# Копируем приложение
COPY app/ .
COPY users.json .
COPY app/.env* .

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]