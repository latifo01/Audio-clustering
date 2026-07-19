FROM python:3.11-slim

WORKDIR /app

# System deps for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Run full pipeline then serve API
CMD ["sh", "-c", "python -m kedro run && uvicorn src.audio_clustering.api.main:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/health || exit 1
