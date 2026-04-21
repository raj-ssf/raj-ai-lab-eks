import json
import logging
import os
from typing import Literal

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("rag-service")

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_FAST = os.environ.get(
    "BEDROCK_FAST_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
MODEL_SMART = os.environ.get(
    "BEDROCK_SMART_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
AUTO_THRESHOLD = int(os.environ.get("AUTO_THRESHOLD_CHARS", "500"))

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

app = FastAPI(title="rag-service", version="0.2.0")


class InvokeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    model: Literal["auto", "fast", "smart"] = "auto"


class InvokeResponse(BaseModel):
    model: str
    routing: Literal["auto", "explicit"]
    text: str
    input_tokens: int
    output_tokens: int


def pick_model(req: InvokeRequest) -> tuple[str, str]:
    """Return (model_id, routing_reason)."""
    if req.model == "fast":
        return MODEL_FAST, "explicit"
    if req.model == "smart":
        return MODEL_SMART, "explicit"
    # auto: length-based heuristic
    if len(req.prompt) < AUTO_THRESHOLD:
        return MODEL_FAST, "auto"
    return MODEL_SMART, "auto"


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {
        "fast": MODEL_FAST,
        "smart": MODEL_SMART,
        "auto_threshold_chars": AUTO_THRESHOLD,
    }


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    model_id, routing = pick_model(req)
    log.info(
        "invoke routing=%s model=%s prompt_chars=%d",
        routing, model_id, len(req.prompt),
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
    except ClientError as e:
        log.error("bedrock invoke failed for model=%s: %s", model_id, e)
        raise HTTPException(status_code=502, detail=str(e))

    payload = json.loads(response["body"].read())
    return InvokeResponse(
        model=model_id,
        routing=routing,
        text=payload["content"][0]["text"],
        input_tokens=payload["usage"]["input_tokens"],
        output_tokens=payload["usage"]["output_tokens"],
    )
