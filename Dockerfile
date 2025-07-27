# --- Этап сборки (Build Stage) ---
FROM python:3.11-slim-buster AS builder

WORKDIR /app

# Устанавливаем зависимости, необходимые для сборки (если есть)
# Например, если torch требует компиляции, здесь могут быть дополнительные пакеты
# RUN apt-get update && apt-get install -y build-essential

# Копируем только requirements.txt для установки зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
# Используем --no-cache-dir для экономии места
# Увеличьте таймаут, если у вас проблемы с сетью
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=120 --retries 5

# --- Этап выполнения (Runtime Stage) ---
FROM python:3.11-slim-buster AS runtime

WORKDIR /app

# Копируем установленные зависимости из этапа сборки
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Копируем остальной код приложения
COPY . .

# Убедитесь, что .dockerignore правильно настроен, чтобы исключить:
# app/parsed_json/
# app/chroma_db/
# .git/
# .venv/

# Открываем порт, который слушает ваше приложение
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]