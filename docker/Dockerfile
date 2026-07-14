# SME Voice Assistant -- Mock Backend
# Runs the FastAPI server in mock mode (no GPU, no model weights needed).
# Image size: ~500MB uncompressed.
#
# Build:
#   docker build -t sme-backend .
#
# Run:
#   docker run -p 5005:5005 sme-backend
#
# Then open: http://localhost:5005/docs

FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only what the backend needs (no checkpoints, no audio files)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.30.1" \
    pydantic==2.7.1 \
    spacy==3.7.4

# Download spaCy model at build time so it is baked into the image
RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY backend.py .

EXPOSE 5005

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "5005"]
