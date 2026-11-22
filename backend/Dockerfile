FROM python:3.12-slim

WORKDIR /app

# System deps for openpyxl & matplotlib (if needed for retraining).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (respects .dockerignore).
COPY . .

# Create a non-root user (UID 1000) and let it own the app + data dirs.
RUN groupadd -g 1000 -r app && useradd -u 1000 -r -g app -d /app -s /sbin/nologin app \
    && mkdir -p /app/data /app/models \
    && chown -R app:app /app

# Runtime config — NO insecure JWT default; the app fails fast if unset.
# Operators MUST set MRTO_JWT_SECRET (>=32 chars) via env/secret.
ENV MRTO_DB_PATH=/app/data/mrto.sqlite3 \
    MRTO_MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Persist DB + models across container rebuilds.
VOLUME ["/app/data", "/app/models"]

# Drop root privileges for the running process.
USER app

EXPOSE 8000

# DB is initialised at container START (entrypoint), not build time,
# so the seeded admin + demo tubes live in the mounted volume, not the
# image layer. This also lets operators rotate the DB by replacing the volume.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

# Graceful shutdown: give in-flight requests up to 30s to finish on SIGTERM.
CMD ["uvicorn", "mrto.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "30"]
