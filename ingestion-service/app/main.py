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
import uuid
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
from pydantic import BaseModel

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
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes": len(body),
        },
    )

    # Phase 2 placeholder — Phase 3 replaces this with the real parse +
    # chunk + embed + write pipeline. Kept as a BackgroundTask so the
    # response shape (202 + immediate return) is already correct for
    # async processing.
    background_tasks.add_task(_process_upload_placeholder, job_id, body, file.content_type)

    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str, _claims: Annotated[dict, Depends(require_jwt)]) -> JobStatus:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


# --- Background processing (placeholder) ------------------------------------

def _process_upload_placeholder(job_id: str, body: bytes, content_type: str) -> None:
    """Phase 2 stub. Logs receipt, marks job done. Phase 3 swaps in the
    real pipeline: Unstructured parse → recursive chunk → bge-m3 embed
    → Qdrant write.
    """
    job = JOBS[job_id]
    job.state = "done"
    job.detail = (
        f"Phase 2 stub — accepted {len(body)} bytes of {content_type}. "
        f"Real ingestion pipeline lands in iteration N+1."
    )
    log.info("upload processed (stub)", extra={"job_id": job_id, "bytes": len(body)})
