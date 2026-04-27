import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal, Optional

import boto3
import httpx
from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from langfuse import Langfuse
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

# --- Phase 4: /retrieve endpoint config ------------------------------------
# /retrieve is the JWT-protected, bge-m3-backed counterpart to /search.
# /search uses Bedrock Titan + the rag-docs collection (legacy path);
# /retrieve uses self-hosted bge-m3 + the documents collection that
# ingestion-service writes to. Two separate embedding spaces — Titan and
# bge-m3 vectors aren't comparable even at the same dim, so we keep the
# collections distinct rather than risk mixing.
EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL", "http://vllm-bge-m3.llm.svc.cluster.local:8000"
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "bge-m3")
DOCS_COLLECTION = os.environ.get("DOCS_COLLECTION", "documents")
RETRIEVE_DEFAULT_TOP_K = int(os.environ.get("RETRIEVE_DEFAULT_TOP_K", "5"))
EMBEDDING_TIMEOUT_SECONDS = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60"))

# Keycloak JWT validation for /retrieve. Issuer-only (audience disabled,
# matching langgraph-service & ingestion-service); the realm is the trust
# boundary — any token signed by it can hit /retrieve, and the in-payload
# user filter scopes results regardless of which client minted the token.
KEYCLOAK_ISSUER = os.environ.get("KEYCLOAK_ISSUER", "")
KEYCLOAK_JWKS_URL = (
    f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs" if KEYCLOAK_ISSUER else ""
)

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

# Langfuse LLM-observability client. Reads LANGFUSE_PUBLIC_KEY,
# LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env. If the keys aren't set, the
# SDK logs a warning and no-ops — every trace call becomes a cheap stub.
# Production deploys should scope the key to a project; ours is one key for
# the lab.
langfuse_client = Langfuse()

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

# Reusable httpx client for the bge-m3 embedding service. Separate from
# vllm_client because connect/read budgets differ — embedding is sub-second
# once warm but cold-start (g6.xlarge JIT scale) can take 3+ min, so we let
# the caller (langgraph-service ensure_warm) handle the warm-up and only
# tolerate short waits here.
embed_client = httpx.Client(
    base_url=EMBEDDING_URL,
    timeout=httpx.Timeout(
        connect=3.0,
        read=EMBEDDING_TIMEOUT_SECONDS,
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


# --- Phase 4: /retrieve models ------------------------------------------
class RetrieveRequest(BaseModel):
    """Per-session retrieval against the bge-m3 / `documents` collection.

    session_id is mandatory: this endpoint is the user-document RAG path
    and crossing sessions would leak documents between chat threads. To
    do an unscoped corpus search, use /search.
    """
    query: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    top_k: int = Field(default=RETRIEVE_DEFAULT_TOP_K, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    count: int
    user: str
    session_id: str


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


# --- JWT validation (used by /retrieve only; legacy endpoints stay open) ---
#
# Lifted from langgraph-service / ingestion-service so all three services
# validate against the same Keycloak realm with consistent semantics.
# Audience is intentionally not verified — Keycloak's default azp/aud
# claims for confidential clients aren't a tight identity boundary in our
# realm, and the user-scoped Qdrant filter below is the actual tenancy
# control.
bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_http_client() -> httpx.Client:
    return httpx.Client(timeout=5.0)


_jwks_cache: dict = {"fetched_at": 0.0, "keys": []}
_JWKS_TTL_SECONDS = 3600


def _fetch_jwks() -> list:
    if not KEYCLOAK_JWKS_URL:
        raise HTTPException(
            status_code=503,
            detail="KEYCLOAK_ISSUER not configured; /retrieve unavailable",
        )
    now = time.time()
    if now - _jwks_cache["fetched_at"] < _JWKS_TTL_SECONDS and _jwks_cache["keys"]:
        return _jwks_cache["keys"]
    log.info("fetching keycloak jwks from %s", KEYCLOAK_JWKS_URL)
    resp = _jwks_http_client().get(KEYCLOAK_JWKS_URL)
    resp.raise_for_status()
    keys = resp.json()["keys"]
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def _decode_jwt(token: str) -> dict:
    unverified = jwt.get_unverified_header(token)
    kid = unverified.get("kid")
    if not kid:
        raise JWTError("token missing kid")
    keys = _fetch_jwks()
    matching = next((k for k in keys if k.get("kid") == kid), None)
    if matching is None:
        # kid rotation — bust cache and refetch once before giving up.
        _jwks_cache["fetched_at"] = 0.0
        keys = _fetch_jwks()
        matching = next((k for k in keys if k.get("kid") == kid), None)
    if matching is None:
        raise JWTError(f"no jwks key matches kid {kid}")
    return jwt.decode(
        token,
        matching,
        algorithms=[matching.get("alg", "RS256")],
        issuer=KEYCLOAK_ISSUER,
        options={"verify_aud": False},
    )


def require_jwt(
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)],
) -> dict:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": 'Bearer realm="rag-service"'},
        )
    try:
        return _decode_jwt(creds.credentials)
    except JWTError as e:
        log.warning("jwt validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {e}",
            headers={"WWW-Authenticate": 'Bearer realm="rag-service"'},
        )


# --- bge-m3 embedding (for /retrieve) ---------------------------------------
#
# Mirrors ingestion-service's _embed_batch shape: vLLM exposes the OpenAI
# /v1/embeddings interface when started with --task=embedding. Single-string
# query, but we still send `input` as a list — vLLM's response always
# carries a `data` array regardless.
def embed_query_bge_m3(text: str) -> list[float]:
    resp = embed_client.post(
        "/v1/embeddings",
        json={"model": EMBEDDING_MODEL_NAME, "input": [text]},
    )
    resp.raise_for_status()
    body = resp.json()
    return body["data"][0]["embedding"]


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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    claims: Annotated[dict, Depends(require_jwt)],
):
    """Phase 4 RAG retrieval.

    Embeds the user's query via vllm-bge-m3 (matching the embedding space
    that ingestion-service writes), then queries the `documents`
    collection filtered by BOTH session_id (from the request body) AND
    user (from the validated JWT claim). The user filter is the
    defense-in-depth check — a caller can guess another session_id, but
    can't forge a different `preferred_username` claim against Keycloak's
    signing keys.
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"

    started = time.monotonic()
    try:
        vec = embed_query_bge_m3(req.query)
    except httpx.HTTPError as e:
        log.error("bge-m3 embed failed: %s", e)
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")

    qfilter = Filter(
        must=[
            FieldCondition(key="session_id", match=MatchValue(value=req.session_id)),
            FieldCondition(key="user",       match=MatchValue(value=user)),
        ]
    )

    try:
        hits = qdrant.query_points(
            collection_name=DOCS_COLLECTION,
            query=vec,
            limit=req.top_k,
            query_filter=qfilter,
        ).points
    except Exception as e:
        # Collection may not exist yet (no uploads in this session). Treat
        # as zero hits rather than 502 — chat-ui still renders a useful
        # "no context found" message.
        log.warning("qdrant query failed (treating as 0 hits): %s", e)
        hits = []

    elapsed_ms = int((time.monotonic() - started) * 1000)
    chunks: list[RetrievedChunk] = []
    for h in hits:
        payload = h.payload or {}
        chunks.append(
            RetrievedChunk(
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                chunk_index=int(payload.get("chunk_index", 0)),
                score=float(h.score),
            )
        )

    log.info(
        "retrieve session=%s user=%s top_k=%d hits=%d ms=%d",
        req.session_id, user, req.top_k, len(chunks), elapsed_ms,
    )

    return RetrieveResponse(
        chunks=chunks,
        count=len(chunks),
        user=user,
        session_id=req.session_id,
    )


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest):
    model_id, routing = pick_model(req)

    # Langfuse trace — one trace per /invoke call. Input is the raw user
    # prompt; retrieval, LLM generation, and final response are attached
    # as child observations below. The trace flushes asynchronously via
    # Langfuse's internal batcher, so this doesn't add request latency.
    trace = langfuse_client.trace(
        name="invoke",
        input={"prompt": req.prompt, "retrieve": req.retrieve, "top_k": req.top_k},
        tags=[f"routing:{routing}", f"provider:{LLM_PROVIDER}"],
    )

    retrieved: Optional[list[RetrievedHit]] = None
    augmented_prompt = req.prompt

    if req.retrieve:
        try:
            retrieved = search_qdrant(req.prompt, req.top_k, req.filter)
        except ClientError as e:
            trace.update(level="ERROR", status_message=f"retrieval failed: {e}")
            raise HTTPException(status_code=502, detail=f"retrieval failed: {e}")

        # Attach retrieval as a span so the Langfuse UI shows it separately
        # from the LLM call. Output is trimmed to ID + score + truncated text
        # so the trace stays readable for prompts that pull long passages.
        trace.span(
            name="retrieve",
            input={"query": req.prompt, "top_k": req.top_k, "filter": req.filter},
            output=[
                {"id": h.id, "score": h.score, "text_preview": (h.text[:200] + "…") if len(h.text) > 200 else h.text}
                for h in retrieved
            ],
        )

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
            trace.update(level="ERROR", status_message=f"vllm: {e}")
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
                trace.update(level="ERROR", status_message=f"both providers failed: {be}")
                raise HTTPException(status_code=502, detail=str(be))

    else:  # "bedrock" (default) or any unrecognized value
        try:
            result = call_bedrock(model_id, augmented_prompt, req.max_tokens)
            provider = "bedrock"
        except ClientError as e:
            log.error("bedrock converse failed for model=%s: %s", model_id, e)
            trace.update(level="ERROR", status_message=str(e))
            raise HTTPException(status_code=502, detail=str(e))

    # Langfuse 'generation' observation — LLM-specific span with token usage,
    # model metadata, and the actual prompt/completion text. This is what
    # makes Langfuse's cost + latency dashboards populate meaningfully.
    trace.generation(
        name=f"{provider}:{result.model_served}",
        model=result.model_served,
        input=augmented_prompt,
        output=result.text,
        usage={
            "input": result.input_tokens,
            "output": result.output_tokens,
        },
        metadata={
            "provider": provider,
            "routing": routing,
        },
    )
    trace.update(output={"text": result.text, "provider": provider, "model": result.model_served})

    return InvokeResponse(
        model=result.model_served,
        routing=routing,
        provider=provider,
        retrieved=retrieved,
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
