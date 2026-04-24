import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, Optional

import boto3
import httpx
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# OpenTelemetry: set up tracer provider + OTLP gRPC exporter BEFORE instrumenting
# boto3/httpx, otherwise the clients are constructed without spans.
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] %(name)s - %(message)s",
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
CHUNK_SIZE_DEFAULT = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP_DEFAULT = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
EMBED_CONCURRENCY = int(os.environ.get("RAG_EMBED_CONCURRENCY", "8"))

# Hybrid routing: which LLM tier serves /invoke.
#   bedrock — always Bedrock (Nova). Lowest latency, managed.
#   vllm    — always vLLM (self-hosted 70B). Bigger model, GPU-backed.
#             Fails loudly if vLLM is unreachable (GPU node group off, etc.).
#   auto    — try vLLM first; on connection error or 5xx, fall back to
#             Bedrock. Cost tradeoff: Bedrock is per-token, vLLM is per-GPU-
#             hour (only billed while the node is up). In "auto" with GPU
#             scaled to zero, connection-refused returns in <1s so the
#             Bedrock fallback adds negligible latency.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock")  # bedrock|vllm|auto
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm.llm.svc.cluster.local:8000")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "llama-3.3-70b")
VLLM_TIMEOUT_SECONDS = float(os.environ.get("VLLM_TIMEOUT_SECONDS", "120"))
VLLM_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("VLLM_CONNECT_TIMEOUT_SECONDS", "3"))

# Payload keys the service manages itself. Excluded from the `metadata` field
# in search results so callers only see what they put there at ingest time.
RESERVED_PAYLOAD_KEYS = {"text", "parent_id", "chunk_index", "original_id"}

# --- OpenTelemetry tracer setup ---
# OTEL_EXPORTER_OTLP_ENDPOINT is read by OTLPSpanExporter automatically, but we
# name the service via Resource so Tempo groups spans correctly in Grafana.
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "rag-service")
_resource = Resource.create({"service.name": SERVICE_NAME})
_provider = TracerProvider(resource=_resource)
_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(_provider)

# Auto-instrument: boto3 (Bedrock calls), httpx (qdrant-client HTTP), stdlib
# logging (injects trace_id/span_id into log records).
BotocoreInstrumentor().instrument()
HTTPXClientInstrumentor().instrument()
LoggingInstrumentor().instrument(set_logging_format=False)

# --- Clients (module-level singletons; constructed AFTER instrumenting) ---
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
qdrant = QdrantClient(url=QDRANT_URL)

# Reusable httpx client for vLLM. Separate connect vs read timeouts: short
# connect lets "auto" fail over quickly when the GPU node is down (connection
# refused returns in <1s); longer read accommodates 70B's tokens-per-second
# on 4x A10G for long completions.
vllm_client = httpx.Client(
    base_url=VLLM_URL,
    timeout=httpx.Timeout(
        connect=VLLM_CONNECT_TIMEOUT_SECONDS,
        read=VLLM_TIMEOUT_SECONDS,
        write=10.0,
        pool=5.0,
    ),
)


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


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Titan embed-v2 has no batch API, so fan out via
    a thread pool. Botocore is thread-safe; OTel botocore instrumentation
    creates one span per call."""
    if not texts:
        return []
    if len(texts) == 1:
        return [embed(texts[0])]
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
        return list(pool.map(embed, texts))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of ~chunk_size chars with `overlap` chars of
    carry-over between neighbours. Prefers paragraph/sentence/word boundaries
    within the last 20% of each chunk so splits don't land mid-word."""
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            search_from = start + int(chunk_size * 0.8)
            for boundary in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(boundary, search_from, end)
                if idx > start:
                    end = idx + len(boundary)
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def build_filter(filter_dict: Optional[dict]) -> Optional[Filter]:
    """Translate a flat dict into a Qdrant Filter with AND semantics. Only
    scalar equality is supported here; range/OR needs a richer surface."""
    if not filter_dict:
        return None
    return Filter(
        must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_dict.items()
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort collection init. If Qdrant is unreachable at startup, log and
    # keep going; the first /ingest call will retry via its own ensure_collection.
    try:
        ensure_collection()
    except Exception as e:
        log.warning("qdrant collection init failed: %s (will retry on first request)", e)
    yield


app = FastAPI(title="rag-service", version="0.6.0", lifespan=lifespan)

# FastAPI trace spans + Prometheus /metrics endpoint.
FastAPIInstrumentor.instrument_app(app)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --- Request / response models ---
class InvokeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    model: Literal["auto", "fast", "smart"] = "auto"
    retrieve: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filter: Optional[dict] = None


class RetrievedHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict = {}
    parent_id: Optional[str] = None
    chunk_index: Optional[int] = None


class InvokeResponse(BaseModel):
    model: str
    routing: Literal["auto", "explicit"]
    # Which tier actually served: bedrock-managed or self-hosted vllm. Callers
    # can use this to attribute cost / latency distributions.
    provider: Literal["bedrock", "vllm"]
    retrieved: Optional[list[RetrievedHit]] = None
    text: str
    input_tokens: int
    output_tokens: int


class IngestRequest(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    metadata: Optional[dict] = None
    chunk_size: int = Field(default=CHUNK_SIZE_DEFAULT, ge=0, le=10_000)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP_DEFAULT, ge=0, le=2_000)


class IngestResponse(BaseModel):
    ids: list[str]
    chunks: int
    chars: int
    parent_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filter: Optional[dict] = None


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


def search_qdrant(
    query: str,
    top_k: int,
    filter_dict: Optional[dict] = None,
) -> list[RetrievedHit]:
    vec = embed(query)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=top_k,
        query_filter=build_filter(filter_dict),
    ).points
    results: list[RetrievedHit] = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            RetrievedHit(
                id=str(h.id),
                score=float(h.score),
                text=payload.get("text", ""),
                metadata={k: v for k, v in payload.items() if k not in RESERVED_PAYLOAD_KEYS},
                parent_id=payload.get("parent_id"),
                chunk_index=payload.get("chunk_index"),
            )
        )
    return results


# --- LLM dispatchers ---
class LLMResult(BaseModel):
    """Provider-agnostic view of an LLM response."""
    text: str
    input_tokens: int
    output_tokens: int
    model_served: str  # model ID / name the provider actually served


def call_bedrock(model_id: str, prompt: str, max_tokens: int) -> LLMResult:
    """Invoke Bedrock Converse API. Raises botocore ClientError on failure."""
    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    return LLMResult(
        text=response["output"]["message"]["content"][0]["text"],
        input_tokens=response["usage"]["inputTokens"],
        output_tokens=response["usage"]["outputTokens"],
        model_served=model_id,
    )


def call_vllm(prompt: str, max_tokens: int) -> LLMResult:
    """Invoke the self-hosted vLLM OpenAI-compatible endpoint. Raises
    httpx.HTTPError on network failure or non-2xx status — callers in 'auto'
    mode catch and fall back to Bedrock."""
    resp = vllm_client.post(
        "/v1/chat/completions",
        json={
            "model": VLLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    body = resp.json()
    return LLMResult(
        text=body["choices"][0]["message"]["content"],
        input_tokens=body["usage"]["prompt_tokens"],
        output_tokens=body["usage"]["completion_tokens"],
        model_served=body.get("model", VLLM_MODEL_NAME),
    )


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
    except Exception as e:
        log.error("ensure_collection failed: %s", e)
        raise HTTPException(status_code=502, detail=f"ensure_collection failed: {e}")

    chunks = chunk_text(req.text, req.chunk_size, req.chunk_overlap)

    try:
        vectors = embed_batch(chunks)
    except ClientError as e:
        log.error("embed failed: %s", e)
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")

    # parent_id is what the caller passed (human-readable). Chunk IDs are
    # uuid5(parent_id:i) so re-ingesting the same input upserts (idempotent)
    # instead of creating duplicates.
    parent_id = req.id or str(uuid.uuid4())

    points: list[PointStruct] = []
    chunk_ids: list[str] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}:{i}"))
        payload = {"text": chunk, "parent_id": parent_id, "chunk_index": i}
        if req.metadata:
            payload.update(req.metadata)
        points.append(PointStruct(id=chunk_id, vector=vec, payload=payload))
        chunk_ids.append(chunk_id)

    qdrant.upsert(collection_name=COLLECTION, points=points)
    log.info("ingested parent=%s chunks=%d chars=%d", parent_id, len(chunks), len(req.text))
    return IngestResponse(
        ids=chunk_ids,
        chunks=len(chunks),
        chars=len(req.text),
        parent_id=parent_id,
    )


@app.post("/search", response_model=list[RetrievedHit])
def search(req: SearchRequest):
    try:
        return search_qdrant(req.query, req.top_k, req.filter)
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    model_id, routing = pick_model(req)

    retrieved: Optional[list[RetrievedHit]] = None
    augmented_prompt = req.prompt

    if req.retrieve:
        try:
            retrieved = search_qdrant(req.prompt, req.top_k, req.filter)
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
        "invoke routing=%s model=%s provider=%s prompt_chars=%d retrieve=%s hits=%d",
        routing, model_id, LLM_PROVIDER, len(req.prompt), req.retrieve,
        len(retrieved) if retrieved else 0,
    )

    # Dispatch based on LLM_PROVIDER. "auto" tries vLLM first and falls back
    # to Bedrock on connection/HTTP errors — this makes the deployment
    # resilient to the GPU node group being scaled to zero between demos.
    result: LLMResult
    provider: Literal["bedrock", "vllm"]

    if LLM_PROVIDER == "vllm":
        try:
            result = call_vllm(augmented_prompt, req.max_tokens)
            provider = "vllm"
        except httpx.HTTPError as e:
            log.error("vllm call failed (provider=vllm, no fallback): %s", e)
            raise HTTPException(status_code=502, detail=f"vllm: {e}")

    elif LLM_PROVIDER == "auto":
        try:
            result = call_vllm(augmented_prompt, req.max_tokens)
            provider = "vllm"
        except httpx.HTTPError as e:
            log.warning("vllm unreachable, falling back to bedrock: %s", e)
            try:
                result = call_bedrock(model_id, augmented_prompt, req.max_tokens)
                provider = "bedrock"
            except ClientError as be:
                log.error("bedrock fallback also failed: %s", be)
                raise HTTPException(status_code=502, detail=str(be))

    else:  # "bedrock" (default) or any unrecognized value
        try:
            result = call_bedrock(model_id, augmented_prompt, req.max_tokens)
            provider = "bedrock"
        except ClientError as e:
            log.error("bedrock converse failed for model=%s: %s", model_id, e)
            raise HTTPException(status_code=502, detail=str(e))

    return InvokeResponse(
        model=result.model_served,
        routing=routing,
        provider=provider,
        retrieved=retrieved,
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
