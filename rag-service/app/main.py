import json
import logging
import os

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
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
)

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

app = FastAPI(title="rag-service", version="0.1.0")


class InvokeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class InvokeResponse(BaseModel):
    model: str
    text: str
    input_tokens: int
    output_tokens: int


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
    except ClientError as e:
        log.error("bedrock invoke failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    payload = json.loads(response["body"].read())
    return InvokeResponse(
        model=MODEL_ID,
        text=payload["content"][0]["text"],
        input_tokens=payload["usage"]["input_tokens"],
        output_tokens=payload["usage"]["output_tokens"],
    )
