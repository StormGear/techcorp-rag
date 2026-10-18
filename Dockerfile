FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY rag-project/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY techcorp-docs /app/techcorp-docs
COPY rag-assistant /app/rag-assistant

WORKDIR /app/rag-assistant

EXPOSE 5252

CMD ["python", "app.py"]
