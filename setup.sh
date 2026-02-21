#!/bin/bash
set -e

echo "🏥 Setting up AyurYukti — आयुर्युक्ति"
echo "Starting infrastructure services..."
docker compose up -d postgres neo4j qdrant ollama

echo "Waiting for services to start..."
sleep 15

echo "Pulling LLM model..."
if docker ps --format '{{.Names}}' | grep -q '^ayuryukti-ollama-1$'; then
  docker exec ayuryukti-ollama-1 ollama pull llama3.1:8b
elif docker ps --format '{{.Names}}' | grep -q '^ayuryukti_ollama_1$'; then
  docker exec ayuryukti_ollama_1 ollama pull llama3.1:8b
else
  OLLAMA_CONTAINER=$(docker ps --format '{{.Names}}' | grep 'ollama' | head -n1)
  docker exec "$OLLAMA_CONTAINER" ollama pull llama3.1:8b
fi

echo "Seeding knowledge base..."
docker compose run --rm ayuryukti python scripts/seed_knowledge_base.py

echo "Starting AyurYukti..."
docker compose up -d ayuryukti

echo "✅ AyurYukti is running at http://localhost:8501"
