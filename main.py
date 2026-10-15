import os
import time
import asyncio
import logging
import httpx
import hmac
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI


GATEWAY_API_KEY = os.environ["GATEWAY_API_KEY"]
VLLM_API_KEY = os.environ["VLLM_API_KEY"]
CALLBACK_URL = os.environ["CALLBACK_URL"]
VLLM_URL = os.environ["VLLM_URL"]
MAX_INFLIGHT = int(os.environ["MAX_INFLIGHT"])
VLLM_TIMEOUT = float(os.environ["VLLM_TIMEOUT"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
 
app = FastAPI()
vllm = AsyncOpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY, timeout=VLLM_TIMEOUT)
http = httpx.AsyncClient(timeout=30)
sem = asyncio.Semaphore(MAX_INFLIGHT)
inflight = {}   # lead_id -> start time
tasks = set()   # keep task refs so they are not garbage collected

OPTIONAL_PARAMS = ("max_tokens", "temperature", "reasoning_effort")
CALLBACK_ATTEMPTS = 5


def check_auth(request):
  auth = request.headers.get("authorization", "")
  if not hmac.compare_digest(auth, "Bearer {}".format(GATEWAY_API_KEY)):
    raise HTTPException(status_code=401, detail="unauthorized")


async def run_inference(lead_id, body):
  payload = {"lead_id": lead_id}
  params = {k: body[k] for k in OPTIONAL_PARAMS if k in body}
  t_accepted = time.time()
  try:
    async with sem:
      t_start = time.time()
      resp = await vllm.chat.completions.create(model=body["model"], messages=body["messages"], **params)
      payload = resp.model_dump()
      payload["lead_id"] = lead_id
      payload["status"] = "ok"
      payload["total_time_s"] = round(time.time() - t_start, 2)
      logger.info("done lead_id=%s total_time_s=%.1fs tokens=%s", lead_id, payload["total_time_s"], resp.usage.completion_tokens if resp.usage else "?")
  except Exception as e:
    logger.exception("inference failed lead_id=%s", lead_id)
    payload["status"] = "error"
    payload["error"] = str(e)
  finally:
    inflight.pop(lead_id, None)
  payload["total_s"] = round(time.time() - t_accepted, 2)
  await send_callback(payload)


async def send_callback(payload):
  lead_id = payload["lead_id"]
  headers = {"Authorization": "Bearer {}".format(GATEWAY_API_KEY)}
  for attempt in range(1, CALLBACK_ATTEMPTS + 1):
    try:
      resp = await http.post(CALLBACK_URL, json=payload, headers=headers)
      if resp.status_code < 300:
        return
      if resp.status_code < 500 and resp.status_code != 429:
        logger.error("Callback rejected lead_id=%s http_status=%s body=%s", lead_id, resp.status_code, resp.text[:500])
        return  # thor said no; the same payload will not do better next time
      logger.warning("Callback failed lead_id=%s attempt=%s http_status=%s", lead_id, attempt, resp.status_code)
    except Exception as e:
      logger.warning("Callback error lead_id=%s attempt=%s error=%s", lead_id, attempt, e)
    if attempt < CALLBACK_ATTEMPTS:
      await asyncio.sleep(2 * attempt)
  logger.error("Callback permanently failed lead_id=%s", lead_id)


@app.post("/inference", status_code=202)
async def infer(request: Request):
  check_auth(request)
  body = await request.json()
  lead_id = body.get("lead_id")
  if not lead_id or not body.get("messages") or not body.get("model"):
    raise HTTPException(status_code=400, detail="lead_id, model and messages are required")
  if lead_id in inflight:
    raise HTTPException(status_code=409, detail="lead_id already in flight")
  inflight[lead_id] = time.time()
  logger.info("Request accepted lead_id=%s inflight=%s", lead_id, len(inflight))
  task = asyncio.create_task(run_inference(lead_id, body))
  tasks.add(task)
  task.add_done_callback(tasks.discard)
  return {"lead_id": lead_id, "status": "accepted"}
 
 
@app.get("/health")
async def health(request: Request):
  check_auth(request)  
  try:
    models = [m.id for m in (await vllm.models.list(timeout=5)).data]
  except Exception as e:
    return JSONResponse({"status": "vllm_not_ready", "error": str(e),}, status_code=503)
  if not models:
    return JSONResponse({"status": "no_model"}, status_code=503)
  return {"status": "ok", "models": models, "inflight": len(inflight)}