FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r ayuryukti && useradd -r -g ayuryukti -d /app -s /sbin/nologin ayuryukti

COPY . .

# Generate synthetic data and train Prakriti model at build time
RUN python scripts/generate_synthetic_data.py

# Create output directories with proper ownership
RUN mkdir -p outputs/ehr outputs/logs outputs/models outputs/reports \
    && chown -R ayuryukti:ayuryukti /app

# Switch to non-root user
USER ayuryukti

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
