Write-Host "Setting up AyurYukti locally (Windows)..." -ForegroundColor Cyan

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/generate_synthetic_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/seed_knowledge_base.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting AyurYukti on http://localhost:8501" -ForegroundColor Green
streamlit run app.py
