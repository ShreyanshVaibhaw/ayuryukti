# AyurYukti Deployment Guide

## Prerequisites

- Docker & Docker Compose v2+
- NVIDIA GPU with drivers (optional, for Ollama LLM)
- 8GB+ RAM recommended
- Python 3.11+ (for local development)

---

## Quick Start (Docker)

### 1. Clone and Configure

```bash
git clone <repository-url>
cd ayuryukti
cp .env.example .env
```

Edit `.env` with secure passwords:
```
POSTGRES_PASSWORD=<your-secure-password>
NEO4J_PASSWORD=<your-secure-password>
JWT_SECRET_KEY=<random-32-char-string>
```

### 2. Start All Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432) — outcome database
- Neo4j (port 7474/7687) — knowledge graph
- Qdrant (port 6333) — vector search
- Ollama (port 11434) — local LLM
- AyurYukti app (port 8501) + API (port 8000)

### 3. Verify Health

```bash
# Check all services
docker-compose ps

# Health endpoints
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### 4. Seed Knowledge Base

```bash
docker-compose exec ayuryukti python scripts/seed_knowledge_base.py
```

### 5. Access

- **Streamlit Dashboard:** http://localhost:8501
- **REST API Docs:** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474

---

## Local Development

### 1. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
python scripts/generate_synthetic_data.py
python scripts/seed_knowledge_base.py
```

### 3. Run Tests

```bash
python -m pytest tests/ -v
```

### 4. Run Benchmarks

```bash
python scripts/benchmark_ner.py
python scripts/benchmark_prakriti.py
python scripts/benchmark_rogaradar.py
```

### 5. Start Application

```bash
# Streamlit dashboard
streamlit run app.py

# FastAPI (separate terminal)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## Production Considerations

### Security
- Change all default passwords in `.env`
- Set `JWT_SECRET_KEY` to a random 32+ character string
- Run behind a reverse proxy (nginx/Traefik) with TLS
- Enable CORS restrictions for production domains

### Performance
- Adjust resource limits in `docker-compose.yml`
- Pre-download Ollama model: `docker-compose exec ollama ollama pull qwen2.5:14b`
- Index formulations in Qdrant for semantic search

### Monitoring
- Health check endpoints: `/health` (API), `/_stcore/health` (Streamlit)
- Docker health checks configured for all services
- Application logs in `outputs/logs/`

### Backup
- PostgreSQL: `docker-compose exec postgres pg_dump -U ayuryukti ayuryukti > backup.sql`
- Neo4j: `docker-compose exec neo4j neo4j-admin database dump neo4j`
