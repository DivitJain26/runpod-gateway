#!/bin/bash
# Installs the wrapper deps and starts the wrapper in the background. vLLM is started by BOOT_CMD after this returns.
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; source .env; set +a; fi
pip install --quiet --root-user-action=ignore -r requirements.txt
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info > /workspace/gateway.log 2>&1 &