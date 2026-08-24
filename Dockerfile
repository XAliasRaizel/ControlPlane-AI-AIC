FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY policies ./policies
COPY README.md .
COPY .env.example .

EXPOSE 8000 8501
