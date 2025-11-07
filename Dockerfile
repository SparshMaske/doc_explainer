# Dockerfile - simple Python image
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r backend/requirements.txt
EXPOSE 8000
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
