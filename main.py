import os
import time
import asyncio
import logging
 
import httpx
from fastapi import FastAPI, Request, HTTPException
from openai import AsyncOpenAI


API_KEY = os.environ["WRAPPER_API_KEY"]          # thor must send: Authorization: Bearer <API_KEY>
CALLBACK_URL = os.environ["CALLBACK_URL"]        # set once; every result is POSTed here
VLLM_URL = os.environ.get("VLLM_URL")
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT"))
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT"))
 
logger = logging.getLogger(__name__)
 
app = FastAPI()
vllm = AsyncOpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=VLLM_TIMEOUT)
http = httpx.AsyncClient(timeout=30)
sem = asyncio.Semaphore(MAX_INFLIGHT)
inflight = {}   # lead_id -> start time
tasks = set()   # keep task refs so they are not garbage collected


def check_auth(request):
  auth = request.headers.get("authorization", "")
  if auth != "Bearer {}".format(API_KEY):
    raise HTTPException(status_code=401, detail="unauthorized")


async def run_inference(lead_id, body):
  payload = {"lead_id": lead_id}
  try:
    async with sem:
      t0 = time.time()
      resp = await vllm.chat.completions.create(
        model=body.get("model"),
        messages=body["messages"],
        max_tokens=body.get("max_tokens"),
        temperature=body.get("temperature"),
        reasoning_effort=body.get("reasoning_effort"),
      )
      payload = resp.model_dump()
      payload["lead_id"] = lead_id
      payload["status"] = "ok"
      logger.info("done lead_id=%s took=%.1fs tokens=%s", lead_id, time.time() - t0, resp.usage.completion_tokens if resp.usage else "?")
  except Exception as e:
    logger.exception("inference failed lead_id=%s", lead_id)
    payload["lead_id"] = lead_id
    payload["status"] = "error"
    payload["error"] = str(e)
  finally:
    inflight.pop(lead_id, None)
  await send_callback(payload)


async def send_callback(payload):
  for attempt in range(3):
    try:
      resp = await http.post(CALLBACK_URL, json=payload, headers={"Authorization": "Bearer {}".format(API_KEY)})
      if resp.status_code < 300:
        return
      logger.error("Callback rejected for lead_id=%s: HTTP %s, response body: %s", payload["lead_id"], resp.status_code, resp.text)
    except Exception:
      logger.exception("Callback request raised an error for lead_id=%s (attempt %s)", payload["lead_id"], attempt)
    await asyncio.sleep(2 * (attempt + 1))
  logger.error("Callback permanently failed for lead_id=%s", payload["lead_id"])


@app.post("/inference", status_code=202)
async def infer(request: Request):
  check_auth(request)
  body = await request.json()
  lead_id = body.get("lead_id")
  if not lead_id or not body.get("messages"):
    raise HTTPException(status_code=400, detail="lead_id and messages are required")
  if lead_id in inflight:
    raise HTTPException(status_code=409, detail="lead_id already in flight")
  inflight[lead_id] = time.time()
  logger.info("Request accepted lead_id=%s inflight=%s", lead_id, len(inflight))
  task = asyncio.create_task(run_inference(lead_id, body))
  tasks.add(task)
  task.add_done_callback(tasks.discard)
  return {"lead_id": lead_id, "status": "accepted"}
 
 
@app.get("/health")
async def health():
  try:
    r = await http.get(VLLM_URL.replace("/v1", "") + "/health", timeout=5)
    vllm_ok = r.status_code == 200
  except Exception:
    vllm_ok = False
  return {"vllm": vllm_ok, "inflight": len(inflight)}