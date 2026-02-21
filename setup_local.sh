#!/bin/bash
set -e

echo "Setting up AyurYukti locally..."
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
python scripts/seed_knowledge_base.py
echo "Starting AyurYukti..."
streamlit run app.py
