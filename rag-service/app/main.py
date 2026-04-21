import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("rag-service")

# --- Config (env-overridable) ---
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_FAST = os.environ.get("BEDROCK_FAST_MODEL_ID", "us.amazon.nova-micro-v1:0")
MODEL_SMART = os.environ.get("BEDROCK_SMART_MODEL_ID", "us.amazon.nova-pro-v1:0")
EMBED_MODEL = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
AUTO_THRESHOLD = int(os.environ.get("AUTO_THRESHOLD_CHARS", "500"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "rag-docs")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
DEFAULT_TOP_K = int(os.environ.get("RAG_DEFAULT_TOP_K", "3"))

# --- Clients (module-level singletons) ---
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    """Idempotently create the Qdrant collection used for RAG."""
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("created collection %s (dim=%d)", COLLECTION, EMBED_DIM)
    else:
        log.info("collection %s already exists", COLLECTION)


def embed(text: str) -> list[float]:
    """Return a Titan embedding for text."""
    body = json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True})
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    return json.loads(response["body"].read())["embedding"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort collection init. If Qdrant is unreachable at startup, log and
    # keep going; the first /ingest call will retry via its own ensure_collection.
    try:
        ensure_collection()
    except Exception as e:
        log.warning("qdrant collection init failed: %s (will retry on first request)", e)
    yield


app = FastAPI(title="rag-service", version="0.4.0", lifespan=lifespan)


# --- Request / response models ---
class InvokeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    model: Literal["auto", "fast", "smart"] = "auto"
    retrieve: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class RetrievedHit(BaseModel):
    id: str
    score: float
    text: str


class InvokeResponse(BaseModel):
    model: str
    routing: Literal["auto", "explicit"]
    retrieved: Optional[list[RetrievedHit]] = None
    text: str
    input_tokens: int
    output_tokens: int


class IngestRequest(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    metadata: Optional[dict] = None


class IngestResponse(BaseModel):
    id: str
    chars: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


# --- Helpers ---
def pick_model(req: InvokeRequest) -> tuple[str, str]:
    """Return (model_id, routing_reason) based on req.model and prompt length."""
    if req.model == "fast":
        return MODEL_FAST, "explicit"
    if req.model == "smart":
        return MODEL_SMART, "explicit"
    if len(req.prompt) < AUTO_THRESHOLD:
        return MODEL_FAST, "auto"
    return MODEL_SMART, "auto"


def search_qdrant(query: str, top_k: int) -> list[RetrievedHit]:
    vec = embed(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
    ).points
    return [
        RetrievedHit(
            id=str(h.id),
            score=float(h.score),
            text=h.payload.get("text", "") if h.payload else "",
        )
        for h in hits
    ]


# --- Endpoints ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {
        "fast": MODEL_FAST,
        "smart": MODEL_SMART,
        "embed": EMBED_MODEL,
        "auto_threshold_chars": AUTO_THRESHOLD,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
        "embed_dim": EMBED_DIM,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        ensure_collection()
        vec = embed(req.text)
    except ClientError as e:
        log.error("embed failed: %s", e)
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")

    # Qdrant point IDs must be unsigned int or UUID. Map arbitrary user-provided
    # IDs to a deterministic UUID5 so callers can still use memorable strings.
    # Original string is preserved in the payload for filtering/lookup.
    payload = {"text": req.text}
    if req.id:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, req.id))
        payload["original_id"] = req.id
    else:
        point_id = str(uuid.uuid4())
    if req.metadata:
        payload.update(req.metadata)

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=vec, payload=payload)],
    )
    log.info("ingested id=%s chars=%d", point_id, len(req.text))
    return IngestResponse(id=point_id, chars=len(req.text))


@app.post("/search", response_model=list[RetrievedHit])
def search(req: SearchRequest):
    try:
        return search_qdrant(req.query, req.top_k)
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    model_id, routing = pick_model(req)

    retrieved: Optional[list[RetrievedHit]] = None
    augmented_prompt = req.prompt

    if req.retrieve:
        try:
            retrieved = search_qdrant(req.prompt, req.top_k)
        except ClientError as e:
            raise HTTPException(status_code=502, detail=f"retrieval failed: {e}")

        if retrieved:
            context = "\n\n".join(
                f"[Source {i + 1}] {h.text}" for i, h in enumerate(retrieved)
            )
            augmented_prompt = (
                "Use the following context to answer the question. If the context "
                "doesn't help, say so.\n\n"
                f"=== Context ===\n{context}\n\n"
                f"=== Question ===\n{req.prompt}"
            )

    log.info(
        "invoke routing=%s model=%s prompt_chars=%d retrieve=%s hits=%d",
        routing, model_id, len(req.prompt), req.retrieve,
        len(retrieved) if retrieved else 0,
    )

    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": augmented_prompt}]}],
            inferenceConfig={"maxTokens": req.max_tokens},
        )
    except ClientError as e:
        log.error("bedrock converse failed for model=%s: %s", model_id, e)
        raise HTTPException(status_code=502, detail=str(e))

    return InvokeResponse(
        model=model_id,
        routing=routing,
        retrieved=retrieved,
        text=response["output"]["message"]["content"][0]["text"],
        input_tokens=response["usage"]["inputTokens"],
        output_tokens=response["usage"]["outputTokens"],
    )
