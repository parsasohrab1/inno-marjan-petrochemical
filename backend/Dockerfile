FROM python:3.12-slim

WORKDIR /app

# System deps for openpyxl & matplotlib (if needed for retraining)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt openpyxl uvicorn[standard]

# Copy application code
COPY . .

# Default env
ENV MRTO_DB_PATH=/app/data/mrto.sqlite3 \
    MRTO_MODELS_DIR=/app/models \
    MRTO_JWT_SECRET=change-me-in-production \
    PYTHONUNBUFFERED=1

# Init DB on container start
RUN mkdir -p /app/data /app/models && python -c "from mrto import db; db.init_db()"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "mrto.app:app", "--host", "0.0.0.0", "--port", "8000"]
