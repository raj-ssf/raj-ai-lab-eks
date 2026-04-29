"""ingestion-service — Phase 2 skeleton.

Endpoints:
  - GET  /healthz              — kubelet probes
  - POST /upload               — accepts file + session_id; validates JWT;
                                 returns 202 with a job_id. Phase 2:
                                 logs receipt, no actual processing.
                                 Phase 3: parses, chunks, embeds, writes
                                 to Qdrant via FastAPI BackgroundTasks.
  - GET  /jobs/{job_id}        — returns job status. Phase 2: in-memory
                                 dict; Phase 3: same dict (single-pod,
                                 no persistence — sufficient for v1).

Auth: Keycloak JWT validation, same shape as langgraph-service. The
JWT is forwarded by chat-ui from the user's browser session and
validated against the realm's JWKs endpoint.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Optional

import httpx
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qmodels

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Match the langgraph-service formatter pattern with defaults for OTel
# fields, so log lines from threads without an active trace context
# (e.g. background tasks) don't crash the formatter.
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] %(name)s - %(message)s",
        defaults={"otelTraceID": "0", "otelSpanID": "0"},
    )
)
_root = logging.getLogger()
_root.handlers = [_handler]
_root.setLevel(logging.INFO)
log = logging.getLogger("ingestion-service")

# --- Config (env-overridable) -----------------------------------------------

KEYCLOAK_ISSUER = os.environ["KEYCLOAK_ISSUER"]
KEYCLOAK_AUDIENCE = os.environ.get("KEYCLOAK_AUDIENCE", "ingestion-service")
KEYCLOAK_JWKS_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"

# Phase 3 wires these up; declared here so the service crashes loudly
# at startup if the Deployment manifest is missing required env.
EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL", "http://vllm-bge-m3.llm.svc.cluster.local:8000"
)
QDRANT_URL = os.environ.get(
    "QDRANT_URL", "http://qdrant.qdrant.svc.cluster.local:6333"
)
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}

# Map content-type → file extension. Unstructured's `partition` dispatches
# to the right parser by file extension, so we have to write the upload
# to a tempfile with the right suffix before parsing.
_EXT_BY_CONTENT_TYPE = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}

# Embedding batch size. vLLM accepts arrays in the /v1/embeddings input
# field; batching reduces round-trip overhead substantially. 32 is a
# good starting point for bge-m3 (chunks are ~1000 chars, well under
# the 8K context — the GPU stays happy with 32 in flight).
EMBEDDING_BATCH_SIZE = 32

# bge-m3's embedding dimension. Used to initialize the Qdrant collection.
# Must match the model — if you swap to nomic-embed-text-v1.5 (768) or
# similar, this constant moves.
EMBEDDING_DIM = 1024

# --- Qdrant client (module-level singleton) ---------------------------------
#
# QdrantClient pools HTTPS connections under the hood, so a single
# client across all background tasks is more efficient than creating
# one per upload.
QDRANT = QdrantClient(url=QDRANT_URL, prefer_grpc=False, https=False)

# --- BM25 sparse encoder (module-level singleton, hybrid search) -------------
#
# Each chunk's text is encoded into BOTH dense (bge-m3 via vLLM) and
# sparse (BM25 via fastembed) vectors. Qdrant stores both under named
# slots in a single point; rag-service /retrieve does a single hybrid
# query that fuses both rankings via Reciprocal Rank Fusion (RRF).
#
# fastembed downloads the BM25 model (~80 MB) on first use to a local
# cache. Pure-Python, runs on CPU, single-shot init (~2 sec).
#
# `passage_embed` is the doc-side encoder; rag-service uses
# `query_embed` for queries (different IDF treatment, recommended by
# the BM25 spec).
BM25 = SparseTextEmbedding(model_name="Qdrant/bm25")


def _ensure_qdrant_collection() -> None:
    """Create the documents collection if it doesn't exist yet.

    Idempotent — safe to call on every upload. Hybrid schema:
      - "dense"  named vector slot, bge-m3's 1024 dims, cosine distance
      - "sparse" named sparse vector slot for BM25 (lexical match)
    Payload schema is implicit (Qdrant is schemaless on payload by
    default); session_id and user are the fields we filter on at
    query time.

    Schema MIGRATION NOTE (post-hybrid-search refactor): this collection
    schema is incompatible with the pre-hybrid (positional dense-only)
    layout. If you find an existing `documents` collection from before
    this change, drop it manually:
      kubectl -n qdrant exec qdrant-0 -- \\
        curl -X DELETE http://localhost:6333/collections/documents
    Then this function recreates it on the next upload, and re-uploads
    populate hybrid vectors.
    """
    if QDRANT.collection_exists(QDRANT_COLLECTION):
        return
    log.info("creating qdrant collection (hybrid)", extra={"collection": QDRANT_COLLECTION})
    QDRANT.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(),
        },
    )
    # Index the session_id payload field so per-session retrieval
    # (rag-service /retrieve filters by session_id) is efficient — without
    # an index, Qdrant scans every payload on every query.
    QDRANT.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="session_id",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )

# --- OTel bootstrap ---------------------------------------------------------

resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "ingestion-service")})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)
LoggingInstrumentor().instrument(set_logging_format=False)
HTTPXClientInstrumentor().instrument()

# --- App + auth -------------------------------------------------------------

app = FastAPI(title="ingestion-service", version="0.1.0")
FastAPIInstrumentor.instrument_app(app)

# Phase #43: Prometheus instrumentation. Same shape as Phase #14a's
# langgraph-service + Phase #24's rag-service: prometheus_fastapi_
# instrumentator wraps the app and exposes /metrics for the
# kube-prometheus-stack ServiceMonitor (Phase #43 ServiceMonitor in
# base/servicemonitor.yaml). Series emitted by default:
#   http_requests_total{method,handler,status}
#   http_request_duration_seconds_bucket{method,handler,le}
#   http_request_size_bytes / http_response_size_bytes
# These let a future Phase wire an AnalysisTemplate gating the
# Phase #39 ingestion-service canary on /upload success rate, in
# place of today's indefinite-pause-and-manual-promote pattern.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks() -> dict:
    """Fetch the realm's JWKs once and cache for the lifetime of the pod.

    Same caching strategy as langgraph-service. Realm key rotation isn't
    handled here — pod restart picks up new keys.
    """
    log.info("fetching keycloak jwks")
    r = httpx.get(KEYCLOAK_JWKS_URL, verify=False, timeout=10.0)
    r.raise_for_status()
    return r.json()


def require_jwt(
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)],
) -> dict:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = creds.credentials
    try:
        unverified = jwt.get_unverified_header(token)
        kid = unverified.get("kid")
        keys = [k for k in _jwks()["keys"] if k.get("kid") == kid]
        if not keys:
            raise HTTPException(status_code=401, detail="kid not in jwks")
        claims = jwt.decode(
            token,
            keys[0],
            algorithms=[unverified.get("alg", "RS256")],
            issuer=KEYCLOAK_ISSUER,
            options={"verify_aud": False},
        )
        return claims
    except JWTError as e:
        log.warning("jwt validation failed: %s", e)
        raise HTTPException(status_code=401, detail=f"invalid token: {e}") from e


# --- Job tracking (in-memory; sufficient for single-pod v1) -----------------

class JobStatus(BaseModel):
    job_id: str
    state: str  # "received" | "parsing" | "chunking" | "embedding" | "writing" | "done" | "failed"
    detail: Optional[str] = None
    chunks_written: int = 0
    session_id: Optional[str] = None
    filename: Optional[str] = None


JOBS: dict[str, JobStatus] = {}


# --- Routes ----------------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True, "service": "ingestion-service", "version": app.version}


@app.post("/upload", status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    claims: Annotated[dict, Depends(require_jwt)],
    file: Annotated[UploadFile, File(...)],
    session_id: Annotated[str, Form(...)],
) -> JobStatus:
    """Accept a file + session_id; enqueue ingestion; return 202 + job_id."""
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content-type: {file.content_type}",
        )

    # Read into memory to enforce size limit. For larger files we'd
    # stream to disk; 25 MiB fits comfortably in the pod's request
    # memory and keeps the code simple.
    body = await file.read()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(body)} > {MAX_UPLOAD_BYTES} bytes)",
        )

    job_id = uuid.uuid4().hex
    job = JobStatus(
        job_id=job_id,
        state="received",
        session_id=session_id,
        filename=file.filename,
    )
    JOBS[job_id] = job
    log.info(
        "upload received",
        extra={
            "job_id": job_id,
            "user": user,
            "session_id": session_id,
            "upload_filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(body),
        },
    )

    # Real ingestion pipeline runs in a background task so /upload
    # returns 202 immediately and the user can poll /jobs/{job_id} for
    # progress.
    background_tasks.add_task(
        _process_upload,
        job_id=job_id,
        body=body,
        content_type=file.content_type,
        filename=file.filename or "unknown",
        session_id=session_id,
        user=user,
    )

    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _claims: Annotated[dict, Depends(require_jwt)]) -> JobStatus:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# --- Background processing --------------------------------------------------

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Call vllm-bge-m3 /v1/embeddings on a batch of strings."""
    with httpx.Client(timeout=120.0, verify=False) as client:
        r = client.post(
            f"{EMBEDDING_URL}/v1/embeddings",
            json={"model": "bge-m3", "input": texts},
        )
        r.raise_for_status()
        return [item["embedding"] for item in r.json()["data"]]


def _process_upload(
    *,
    job_id: str,
    body: bytes,
    content_type: str,
    filename: str,
    session_id: str,
    user: str,
) -> None:
    """The real ingestion pipeline.

    parse → chunk → embed → write to Qdrant. Updates JOBS[job_id].state
    at each transition so /jobs/{id} reports useful progress.
    Exceptions are caught and surfaced via job.state='failed' + detail.
    """
    job = JOBS[job_id]
    file_path: Optional[str] = None

    try:
        # Heavy imports deferred to first call so pod cold-start stays
        # quick (Unstructured pulls a lot of optional deps).
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from unstructured.partition.auto import partition

        # 1. Save bytes to disk with the right extension so partition()
        #    can dispatch to the correct parser.
        suffix = _EXT_BY_CONTENT_TYPE.get(content_type, ".bin")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(body)
            file_path = f.name
        log.info(
            "upload buffered to disk",
            extra={"job_id": job_id, "path": file_path, "bytes": len(body)},
        )

        # 2. Parse via Unstructured.
        job.state = "parsing"
        log.info("parsing", extra={"job_id": job_id, "upload_filename": filename})
        elements = partition(filename=file_path)
        raw_text = "\n\n".join(str(el) for el in elements if str(el).strip())
        if not raw_text:
            job.state = "done"
            job.detail = "no extractable text found in document"
            log.info("upload done (no text)", extra={"job_id": job_id})
            return

        # 3. Chunk. RecursiveCharacterTextSplitter prefers paragraph →
        # sentence → word breaks before falling back to arbitrary cuts.
        # 1000 chars / 200 overlap is the standard RAG sweet spot.
        job.state = "chunking"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_text(raw_text)
        log.info(
            "chunked",
            extra={"job_id": job_id, "chunks": len(chunks), "raw_chars": len(raw_text)},
        )
        if not chunks:
            job.state = "done"
            job.detail = "chunking produced no output"
            return

        # 4. Embed in batches against vllm-bge-m3.
        job.state = "embedding"
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            batch_embeds = _embed_batch(batch)
            embeddings.extend(batch_embeds)
            log.info(
                "embedded batch",
                extra={
                    "job_id": job_id,
                    "from": i,
                    "to": i + len(batch),
                    "total": len(chunks),
                },
            )

        # 5. Compute BM25 sparse vectors for the same chunks. Cheap
        # (CPU only, ~ms per chunk) and runs sequentially in this
        # background task — no need to batch at the network layer
        # since fastembed runs in-process. The result is one
        # SparseEmbedding per chunk, which we convert to Qdrant's
        # SparseVector wire shape below.
        job.state = "encoding-sparse"
        sparse_vectors = list(BM25.passage_embed(chunks))

        # 6. Write to Qdrant. Lazy collection creation — first upload
        # initializes it (hybrid schema with dense + sparse named slots),
        # subsequent uploads see it exists and skip.
        job.state = "writing"
        _ensure_qdrant_collection()
        ingested_at = datetime.utcnow().isoformat() + "Z"
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                # Vector as a dict of named slots (the hybrid schema
                # declares both "dense" and "sparse"). Mismatched names
                # cause a 400 from Qdrant at upsert time, so this dict
                # has to match _ensure_qdrant_collection's vectors_config
                # + sparse_vectors_config exactly.
                vector={
                    "dense": embeddings[i],
                    "sparse": qmodels.SparseVector(
                        indices=sparse_vectors[i].indices.tolist(),
                        values=sparse_vectors[i].values.tolist(),
                    ),
                },
                payload={
                    "text": chunks[i],
                    "source": filename,
                    "chunk_index": i,
                    "session_id": session_id,
                    "user": user,
                    "ingested_at": ingested_at,
                },
            )
            for i in range(len(chunks))
        ]
        QDRANT.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

        job.state = "done"
        job.chunks_written = len(chunks)
        log.info(
            "upload ingestion complete",
            extra={
                "job_id": job_id,
                "upload_filename": filename,
                "chunks": len(chunks),
                "session_id": session_id,
            },
        )

    except Exception as e:
        log.exception("upload ingestion failed: %s", e)
        job.state = "failed"
        # Truncate the detail to avoid leaking long stack traces in API
        # responses; full traceback lives in pod logs for debugging.
        job.detail = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        if file_path:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
