FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/
ENV PYTHONPATH=/app/src
EXPOSE 8000 8501
CMD ["uvicorn", "docrouter.api:app", "--host", "0.0.0.0", "--port", "8000"]
