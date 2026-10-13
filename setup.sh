#!/bin/bash
# One-time setup on the pod. Run from /workspace after copying the files in:
#   bash /workspace/setup.sh
set -e
cd /workspace

if [ ! -f .env ]; then
  echo ".env missing — copy .env.example to .env and fill it in"; exit 1
fi
set -a; source .env; set +a
if [ "$WRAPPER_API_KEY" = "change-me" ] || [ -z "$CALLBACK_URL" ]; then
  echo "fill in WRAPPER_API_KEY and CALLBACK_URL in .env"; exit 1
fi

# vLLM's image already ships fastapi/uvicorn/openai/httpx; pin them anyway so
# the wrapper doesn't break when the base image moves.
pip install --quiet -r requirements.txt

python -c "import fastapi, uvicorn, openai, httpx; print('wrapper deps ok')"
chmod +x start.sh
echo "setup done — set the RunPod container start command to: bash /workspace/start.sh"