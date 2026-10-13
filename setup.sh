#!/bin/bash
# Installs the wrapper deps and starts the wrapper. vLLM is started by the image itself.
set -e
set -a; source .env; set +a
pip install --quiet -r requirements.txt
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info