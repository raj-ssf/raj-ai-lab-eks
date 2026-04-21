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
    "amazon.nova-micro-v1:0",
)
MODEL_SMART = os.environ.get(
    "BEDROCK_SMART_MODEL_ID",
    "amazon.nova-pro-v1:0",
)
AUTO_THRESHOLD = int(os.environ.get("AUTO_THRESHOLD_CHARS", "500"))

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

app = FastAPI(title="rag-service", version="0.3.1")


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
    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": req.prompt}]}],
            inferenceConfig={"maxTokens": req.max_tokens},
        )
    except ClientError as e:
        log.error("bedrock converse failed for model=%s: %s", model_id, e)
        raise HTTPException(status_code=502, detail=str(e))

    return InvokeResponse(
        model=model_id,
        routing=routing,
        text=response["output"]["message"]["content"][0]["text"],
        input_tokens=response["usage"]["inputTokens"],
        output_tokens=response["usage"]["outputTokens"],
    )
