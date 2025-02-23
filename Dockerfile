# Stage 1: Build and generate cache files
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN py -m news_classifier.run --config-file config/random_forest.json

# Stage 2: Production image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app .

EXPOSE 8000

CMD ["uvicorn", "news_classifier.model_integration.api:app", "--host", "0.0.0.0", "--port", "8000"]
