FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --default-timeout=1000 --retries 10 -r requirements.txt

COPY . .

# Default command (Docker Compose isko override kar lega)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]