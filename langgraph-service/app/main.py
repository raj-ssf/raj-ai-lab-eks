"""langgraph-service — FastAPI front-end for a LangGraph router agent.

Three-node state machine on /invoke:

    START -> classify -> ensure_warm -> execute -> END

- classify: always-on Llama 3.1 8B answers a JSON-schema classification
  prompt categorizing the user's input as `trivial` / `reasoning` / `hard`.
- ensure_warm: if the routed-to variant is at replicas=0, scale to 1 and
  wait for Ready. Iteration 2c content; currently a passthrough.
- execute: HTTP POST to the chosen variant's vLLM OpenAI-compat endpoint.

Every node emits a Langfuse span via the LangChain callback handler. The
whole flow is also instrumented for OTel so traces appear in Tempo.

Auth: /invoke is bearer-token protected (Keycloak JWT). /healthz is open
for kubelet probes.
"""

import logging
import os
import time
from functools import lru_cache
from typing import Annotated, Literal, Optional, TypedDict

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler as LangfuseCallback
from langgraph.graph import END, START, StateGraph
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Use Formatter `defaults=` (Python 3.10+) so log records emitted from
# threads without an active OTel trace context (e.g. the OTel exporter's
# own background error logging) don't crash the formatter with
# 'Formatting field not found in record: otelTraceID'. The "0" sentinel
# matches OTel's no-trace convention.
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
log = logging.getLogger("langgraph-service")

# --- Config (env-overridable) -----------------------------------------------

KEYCLOAK_ISSUER = os.environ["KEYCLOAK_ISSUER"]
KEYCLOAK_AUDIENCE = os.environ.get("KEYCLOAK_AUDIENCE", "langgraph-service")
KEYCLOAK_JWKS_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"

# Routing destinations. Each value is the base URL of the OpenAI-compat
# vLLM Service for that tier; we append /v1/chat/completions at call time.
MODEL_TRIVIAL_URL = os.environ.get(
    "MODEL_TRIVIAL_URL",
    "http://vllm-llama-8b.llm.svc.cluster.local:8000",
)
MODEL_REASONING_URL = os.environ.get(
    "MODEL_REASONING_URL",
    "http://vllm-deepseek-r1-70b.llm.svc.cluster.local:8000",
)
MODEL_HARD_URL = os.environ.get(
    "MODEL_HARD_URL",
    "http://vllm.llm.svc.cluster.local:8000",
)

# --- vLLM model names (the `model` field that vLLM's --served-model-name
# flag declares). These come from the variant Deployments' args; if those
# args change, these need to follow.
MODEL_TRIVIAL_NAME = os.environ.get("MODEL_TRIVIAL_NAME", "llama-3.1-8b")
MODEL_REASONING_NAME = os.environ.get("MODEL_REASONING_NAME", "deepseek-r1-distill-llama-70b")
MODEL_HARD_NAME = os.environ.get("MODEL_HARD_NAME", "llama-3.3-70b")
# Fine-tuning F4.5: the Alpaca-tuned LoRA adapter is served by the SAME
# vllm-llama-8b pod that hosts the trivial-tier base model. vLLM merges
# the adapter in flight when called with this model name (see F3:
# --enable-lora --lora-modules llama-3.1-8b-alpaca=/adapters/alpaca-lora).
# So tuned-lora reuses MODEL_TRIVIAL_URL and DEPLOY_TRIVIAL — no separate
# Deployment, no separate scaling. Effectively a "stylistic variant" of
# the trivial tier, picked when verbose/instruction-style answers fit.
MODEL_TUNED_LORA_NAME = os.environ.get("MODEL_TUNED_LORA_NAME", "llama-3.1-8b-alpaca")

# Deployment names for the JIT-scale path (iteration 2c).
DEPLOY_TRIVIAL = os.environ.get("DEPLOY_TRIVIAL", "vllm-llama-8b")
DEPLOY_REASONING = os.environ.get("DEPLOY_REASONING", "vllm-deepseek-r1-70b")
DEPLOY_HARD = os.environ.get("DEPLOY_HARD", "vllm")
LLM_NAMESPACE = os.environ.get("LLM_NAMESPACE", "llm")

# Inference parameters. The classifier prompt benefits from low temperature
# (deterministic output). Execute calls inherit the user's max_tokens.
CLASSIFIER_TEMPERATURE = 0.0
CLASSIFIER_MAX_TOKENS = 32
CLASSIFIER_TIMEOUT_SECONDS = 15
EXECUTE_TIMEOUT_SECONDS = 120

# --- Phase 4: RAG retrieve node config -------------------------------------
# rag-service /retrieve is JWT-protected (validates Keycloak issuer). We
# forward the same user JWT we received on /invoke; rag-service uses its
# `preferred_username` claim as the second axis of the Qdrant filter
# (the first being session_id). If session_id isn't set on the
# InvokeRequest, the retrieve node short-circuits — the chat-ui hook
# that supplies it lands in Phase 5.
RAG_SERVICE_URL = os.environ.get(
    "RAG_SERVICE_URL", "http://rag-service.rag.svc.cluster.local"
)
RAG_RETRIEVE_TIMEOUT_SECONDS = float(os.environ.get("RAG_RETRIEVE_TIMEOUT_SECONDS", "30"))
RAG_TOP_K_DEFAULT = int(os.environ.get("RAG_TOP_K_DEFAULT", "5"))

# JIT scale-up parameters. Karpenter's GPU node typically takes 60-90s to
# come up; vLLM cold-start adds another 2-3 min for 70B AWQ. 5 min total
# is a comfortable upper bound; bump for the 405B path (which can take 6+
# min for S3-Mountpoint weight fault-in).
# 15 min covers a full 70B cold start: Karpenter provision (~90s) +
# node boot (~60s) + image pull (~5s cached / ~3min cold) + EBS attach
# (~20s) + model-sync s3 sync (~60s) + vLLM safetensors load (~5min on
# 4× PCIe L4) + KV cache warmup (~30s) + sidecar handshake (~10s).
# 405B via S3 Mountpoint can stretch to ~10 min for fault-in.
SCALE_UP_TIMEOUT_SECONDS = 900
SCALE_POLL_INTERVAL_SECONDS = 3

# --- Tool registry: route name → call config -------------------------------

ROUTE_REGISTRY: dict[str, dict] = {
    "trivial": {
        "url": f"{MODEL_TRIVIAL_URL}/v1",
        "model_name": MODEL_TRIVIAL_NAME,
        "deployment": DEPLOY_TRIVIAL,
        "always_on": True,  # Llama 8B is the always-on tier; never JIT-scaled
        # Tool calling: vllm-llama-8b is started with
        # --enable-auto-tool-choice + --tool-call-parser llama3_json
        # (see llm/base/deployment-models.yaml). The Llama 3.1 chat
        # template natively encodes tool definitions, so the model
        # returns structured tool_calls when it decides to use one.
        # Routes WITHOUT this flag take the legacy single-shot path
        # in node_execute — current 70B / DeepSeek deployments don't
        # have the tool-call-parser flag wired up yet.
        "supports_tools": True,
    },
    "tuned-lora": {
        # Reuses the trivial tier's vLLM pod and Deployment — the
        # llama-3.1-8b-alpaca model name resolves to the LoRA adapter
        # merged on top of the same Llama 3.1 8B base. always_on=True
        # because there's no second pod to JIT-scale; if the trivial
        # tier is up, this tier is up too.
        "url": f"{MODEL_TRIVIAL_URL}/v1",
        "model_name": MODEL_TUNED_LORA_NAME,
        "deployment": DEPLOY_TRIVIAL,
        "always_on": True,
        # Tools work for the LoRA-merged model too — the adapter
        # modifies attention/projection weights but keeps the chat
        # template's tool-call grammar intact.
        "supports_tools": True,
    },
    "reasoning": {
        "url": f"{MODEL_REASONING_URL}/v1",
        "model_name": MODEL_REASONING_NAME,
        "deployment": DEPLOY_REASONING,
        "always_on": False,
        "supports_tools": False,  # DeepSeek-R1 70B's chat-template support TBD
    },
    "hard": {
        "url": f"{MODEL_HARD_URL}/v1",
        "model_name": MODEL_HARD_NAME,
        "deployment": DEPLOY_HARD,
        "always_on": False,
        "supports_tools": False,  # 70B AWQ tier; tool support TBD
    },
}

# --- Kubernetes client (in-cluster config) ----------------------------------

# Loaded once at import time. Reads the projected serviceaccount token + CA
# cert from /var/run/secrets/kubernetes.io/serviceaccount/. The
# langgraph-service ServiceAccount is bound (via the Role+RoleBinding in
# raj-ai-lab-eks/langgraph-service/base/serviceaccount.yaml) to scale
# Deployments in the llm namespace — that's the only privilege we need.
try:
    k8s_config.load_incluster_config()
    K8S_APPS = k8s_client.AppsV1Api()
    K8S_CORE = k8s_client.CoreV1Api()
    log.info("kubernetes in-cluster config loaded")
except k8s_config.ConfigException as e:
    # Allow local dev / unit tests without an in-cluster context. The JIT
    # scale path will raise at first use rather than at import; healthz
    # and /invoke without scaling-required routes still work.
    log.warning("kubernetes in-cluster config unavailable: %s", e)
    K8S_APPS = None  # type: ignore[assignment]
    K8S_CORE = None  # type: ignore[assignment]


def _scale_deployment(name: str, replicas: int) -> None:
    """Patch /scale on a Deployment in the llm namespace.

    Uses the scale subresource (not full deployment patch) — the SA's Role
    only grants `apps/deployments/scale`, so any attempt to mutate other
    fields would 403.
    """
    if K8S_APPS is None:
        raise RuntimeError("kubernetes client not initialized")
    body = {"spec": {"replicas": replicas}}
    K8S_APPS.patch_namespaced_deployment_scale(name=name, namespace=LLM_NAMESPACE, body=body)


def _read_replicas(name: str) -> int:
    """Read the current spec.replicas of a Deployment via /scale subresource."""
    if K8S_APPS is None:
        raise RuntimeError("kubernetes client not initialized")
    scale = K8S_APPS.read_namespaced_deployment_scale(name=name, namespace=LLM_NAMESPACE)
    return scale.spec.replicas or 0


def _read_available_replicas(name: str) -> int:
    """Read status.availableReplicas — pods actually passing readinessProbe.

    Different from spec.replicas (desired). A Deployment can have
    spec.replicas=1 and availableReplicas=0 for the entire cold-start
    window (Karpenter + image pull + model load). ensure_warm must use
    THIS to decide whether to short-circuit; reading spec.replicas
    risks declaring 'already warm' for a pod still in Init:2/3.
    """
    if K8S_APPS is None:
        raise RuntimeError("kubernetes client not initialized")
    deploy = K8S_APPS.read_namespaced_deployment(name=name, namespace=LLM_NAMESPACE)
    return deploy.status.available_replicas or 0


def _wait_for_pod_ready(label_selector: str, timeout_s: int) -> bool:
    """Poll until at least one pod matching the selector is Ready, or timeout.

    Returns True if a Ready pod appeared within the timeout, False if not.
    Polling (rather than a Watch) is intentional here: the request rate is
    one-per-cold-start and the simple polling code is more debuggable than
    a Watch's resourceVersion / event-stream complexity.
    """
    if K8S_CORE is None:
        raise RuntimeError("kubernetes client not initialized")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pods = K8S_CORE.list_namespaced_pod(
            namespace=LLM_NAMESPACE,
            label_selector=label_selector,
        )
        for pod in pods.items:
            for cond in (pod.status.conditions or []):
                if cond.type == "Ready" and cond.status == "True":
                    return True
        time.sleep(SCALE_POLL_INTERVAL_SECONDS)
    return False


# --- OTel bootstrap ---------------------------------------------------------

resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "langgraph-service")})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(tracer_provider)
LoggingInstrumentor().instrument(set_logging_format=False)
HTTPXClientInstrumentor().instrument()
tracer = trace.get_tracer(__name__)

# --- Langfuse callback ------------------------------------------------------

# v3 SDK changed the integration shape: the CallbackHandler is constructed
# once (no per-request user_id / tags kwargs anymore) and reads
# LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env. Per-
# request attribution (user, session, tags) is passed through LangChain's
# config metadata instead — see the /invoke handler.
_LANGFUSE_CB: Optional[LangfuseCallback] = (
    LangfuseCallback()
    if (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
    else None
)


# --- App + auth -------------------------------------------------------------

app = FastAPI(title="langgraph-service", version="0.2.0")
FastAPIInstrumentor.instrument_app(app)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _jwks_client() -> httpx.Client:
    return httpx.Client(timeout=5.0)


_jwks_cache: dict = {"fetched_at": 0.0, "keys": []}
_JWKS_TTL_SECONDS = 3600


def _fetch_jwks() -> list:
    now = time.time()
    if now - _jwks_cache["fetched_at"] < _JWKS_TTL_SECONDS and _jwks_cache["keys"]:
        return _jwks_cache["keys"]
    log.info("fetching keycloak jwks", extra={"jwks_url": KEYCLOAK_JWKS_URL})
    resp = _jwks_client().get(KEYCLOAK_JWKS_URL)
    resp.raise_for_status()
    keys = resp.json()["keys"]
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    return keys


def _decode_jwt(token: str) -> dict:
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise JWTError("token missing kid in header")
    keys = _fetch_jwks()
    matching = next((k for k in keys if k.get("kid") == kid), None)
    if matching is None:
        _jwks_cache["fetched_at"] = 0
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
            headers={"WWW-Authenticate": 'Bearer realm="langgraph-service"'},
        )
    try:
        return _decode_jwt(creds.credentials)
    except JWTError as e:
        log.warning("jwt validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {e}",
            headers={"WWW-Authenticate": 'Bearer realm="langgraph-service"'},
        )


# --- LangGraph state machine ------------------------------------------------

class AgentState(TypedDict, total=False):
    """Graph state passed between nodes.

    LangGraph merges dict updates between nodes — each node returns a
    partial state and the graph reducer merges them into the running
    state. Using TypedDict (rather than Pydantic) is the LangGraph-native
    pattern and avoids per-update revalidation overhead.
    """
    prompt: str
    max_tokens: int
    image_url: Optional[str]
    user: str
    # Set by /invoke handler before invoke; forwarded into retrieve node
    session_id: Optional[str]
    top_k: int
    # The user's bearer token, forwarded to rag-service /retrieve so it
    # can validate against Keycloak and read the user claim. NOT logged.
    auth_token: Optional[str]
    # Set by classify
    route: Literal["trivial", "tuned-lora", "reasoning", "hard"]
    classifier_raw: str
    # Set by retrieve (Phase 4)
    retrieved_chunks: list[dict]   # list of {text, source, chunk_index, score}
    retrieve_count: int
    retrieve_ms: int
    # Set by ensure_warm
    cold_start: bool
    warm_wait_seconds: float
    # Set by execute
    response: str
    execute_latency_ms: int
    # Tool calling (Phase T) — populated by node_execute when the
    # routed-to model supports tools and the agent loop fires off
    # tool calls. tool_iterations counts loop passes (capped at
    # AGENT_MAX_ITERATIONS); tool_calls_log is the ordered list of
    # tools the agent invoked, useful for chat-ui display + Langfuse
    # trace correlation.
    tool_iterations: int
    tool_calls_log: list[str]


# Classifier system prompt — short, direct, JSON-output. The trick is to
# keep this prompt small enough that the always-on Llama 8B can run it
# fast (~200ms typical) and the routing decision dwarfs end-to-end
# latency only on cold starts.
CLASSIFIER_SYSTEM_PROMPT = """You are a routing classifier. Read the user's prompt and output exactly one word from this set:

- trivial: single-fact recall, single-step arithmetic, yes/no, short factual lookup. Answer fits in one sentence and needs no working out.
- reasoning: multi-step thinking, chain-of-thought, math WORD problems (anything where the user describes a scenario and asks you to compute the answer through several steps), logic puzzles, deduction. If the prompt requires combining 2+ facts or doing 2+ calculations, this is reasoning — NOT trivial.
- tuned-lora: instruction-following with explanation; "how do I...", "explain...", "what is the difference between...", tutorial-style requests where a verbose, structured answer is expected. Use this for explanations of concepts, NOT for math/logic problems.
- hard: complex tasks (long-form writing, code generation, deep analysis, multi-paragraph essays)

Examples:
  Prompt: "What's the capital of France?" -> trivial
  Prompt: "What is 2+2?" -> trivial
  Prompt: "If a train leaves Chicago at 3pm going 60mph and another leaves NYC at 4pm going 80mph, when do they meet?" -> reasoning
  Prompt: "If A implies B and B implies C, does A imply C?" -> reasoning
  Prompt: "Explain how Kubernetes pods work." -> tuned-lora
  Prompt: "How do I write a Python decorator?" -> tuned-lora
  Prompt: "Write a 1000-word essay comparing capitalism and socialism." -> hard
  Prompt: "Implement a thread-safe LRU cache in Go." -> hard

Output ONLY the single word. No explanation. No punctuation."""


def node_classify(state: AgentState) -> AgentState:
    """Use the always-on Llama 8B to label the prompt's tier."""
    cfg = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg["model_name"],
        base_url=cfg["url"],
        api_key="not-required",  # vLLM doesn't enforce; required by the lib
        temperature=CLASSIFIER_TEMPERATURE,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        timeout=CLASSIFIER_TIMEOUT_SECONDS,
    )
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=state["prompt"]),
    ]
    response = client.invoke(messages)
    raw = (response.content or "").strip().lower()

    # Hardened parsing — model occasionally adds punctuation despite the
    # instruction. Fall back to "trivial" on any unparseable response so
    # the cheap path handles ambiguous cases. Order matters for substring
    # matching: check the most specific names first ("tuned-lora" before
    # "reasoning"/"trivial"; "reasoning" before "trivial") to avoid the
    # 'r' in "trivial" accidentally satisfying a "tuned-lora" target etc.
    route: Literal["trivial", "tuned-lora", "reasoning", "hard"] = "trivial"
    for candidate in ("tuned-lora", "hard", "reasoning", "trivial"):
        if candidate in raw:
            route = candidate  # type: ignore[assignment]
            break

    # Inline the route + raw output in the message itself: the existing
    # formatter doesn't surface log-record `extra` fields, so the previous
    # `extra={...}` form left these values invisible in pod logs. Inline
    # makes them grep-able for routing diagnostics.
    log.info("classified route=%s classifier_raw=%r", route, raw[:80])
    return {"route": route, "classifier_raw": raw}


def node_retrieve(state: AgentState) -> AgentState:
    """Phase 4: per-session RAG retrieval.

    Calls rag-service POST /retrieve with the user's prompt + session_id.
    rag-service embeds via vllm-bge-m3 and queries the `documents`
    Qdrant collection filtered by (session_id AND user-from-JWT). The
    returned chunks land in state and are prepended to the prompt by
    node_execute as RAG context.

    Skips when:
      - session_id is None or empty (no chat session — chat-ui upload
        flow hasn't run; nothing to retrieve)
      - rag-service is unreachable (logged + treated as zero hits; we
        don't want a flaky retrieve to break /invoke entirely)
    """
    session_id = state.get("session_id")
    if not session_id:
        return {"retrieved_chunks": [], "retrieve_count": 0, "retrieve_ms": 0}

    auth_token = state.get("auth_token")
    if not auth_token:
        # Should never happen — /invoke handler always sets it when
        # session_id is set. Defensive log + skip rather than 500.
        log.warning("retrieve: session_id set but auth_token missing; skipping")
        return {"retrieved_chunks": [], "retrieve_count": 0, "retrieve_ms": 0}

    started = time.monotonic()
    chunks: list[dict] = []
    try:
        with httpx.Client(timeout=RAG_RETRIEVE_TIMEOUT_SECONDS) as client:
            resp = client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={
                    "query": state["prompt"],
                    "session_id": session_id,
                    "top_k": state.get("top_k", RAG_TOP_K_DEFAULT),
                },
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            resp.raise_for_status()
            body = resp.json()
            chunks = body.get("chunks", [])
    except httpx.HTTPError as e:
        # Don't fail the request — RAG is enrichment, not the critical
        # path. Log + return empty so execute proceeds without context.
        log.warning("retrieve failed (continuing without context): %s", e)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "retrieve",
        extra={
            "session_id": session_id,
            "chunks": len(chunks),
            "retrieve_ms": elapsed_ms,
        },
    )
    return {
        "retrieved_chunks": chunks,
        "retrieve_count": len(chunks),
        "retrieve_ms": elapsed_ms,
    }


def node_ensure_warm(state: AgentState) -> AgentState:
    """JIT-scale the chosen variant if it's cold, then wait for Ready.

    Steady-state behavior:
      - trivial route (always_on=True): no-op, return cold_start=False
      - heavier routes: read current replicas. If >=1, no-op (variant is
        already warm). If 0, patch scale to 1, wait for Ready, return
        cold_start=True with elapsed seconds.

    On scale-up failure (deployment missing, RBAC denied) or wait-for-ready
    timeout (Karpenter couldn't provision GPU, image pull stuck, etc.),
    we raise — LangGraph propagates the exception and the FastAPI layer
    converts it to a 502 with the underlying message.
    """
    cfg = ROUTE_REGISTRY[state["route"]]
    if cfg["always_on"]:
        return {"cold_start": False, "warm_wait_seconds": 0.0}

    deploy = cfg["deployment"]
    log.info("ensure_warm checking", extra={"deployment": deploy, "route": state["route"]})

    started = time.monotonic()
    try:
        spec_replicas = _read_replicas(deploy)
        available = _read_available_replicas(deploy)
    except ApiException as e:
        raise RuntimeError(f"failed to read scale of {LLM_NAMESPACE}/{deploy}: {e.reason}") from e

    # Three states to handle:
    #   (1) available >= 1                 → variant is warm + serving. Skip.
    #   (2) spec >= 1 but available < 1    → cold-start in progress (pod
    #       exists but still in Init / model-load). Don't re-scale; wait.
    #   (3) spec == 0                      → not scaled yet. Scale, then wait.
    #
    # The previous version checked only spec.replicas and returned
    # immediately on (2). That caused execute to fire against a Service
    # with no Ready endpoints → 'no healthy upstream' → 500.

    if available >= 1:
        log.info(
            "ensure_warm: variant warm + serving, skipping wait",
            extra={"deployment": deploy, "available_replicas": available},
        )
        return {"cold_start": False, "warm_wait_seconds": 0.0}

    if spec_replicas == 0:
        log.info("ensure_warm: scaling up cold variant", extra={"deployment": deploy})
        try:
            _scale_deployment(deploy, replicas=1)
        except ApiException as e:
            raise RuntimeError(f"failed to scale {LLM_NAMESPACE}/{deploy}: {e.reason}") from e
    else:
        log.info(
            "ensure_warm: variant cold-starting, waiting for ready",
            extra={"deployment": deploy, "spec_replicas": spec_replicas, "available_replicas": available},
        )

    # The variant Deployments label their pods with `app: <deployment-name>`
    # — match deployment-models.yaml in the app repo. If that scheme ever
    # changes, this selector must follow.
    label_selector = f"app={deploy}"
    ready = _wait_for_pod_ready(label_selector, SCALE_UP_TIMEOUT_SECONDS)
    elapsed = time.monotonic() - started
    if not ready:
        raise RuntimeError(
            f"{deploy} did not become Ready within {SCALE_UP_TIMEOUT_SECONDS}s "
            f"(check Karpenter NodeClaim status + the variant pod's Events)"
        )

    log.info(
        "ensure_warm: warmed",
        extra={"deployment": deploy, "warm_wait_seconds": round(elapsed, 1)},
    )
    return {"cold_start": True, "warm_wait_seconds": elapsed}


def _build_rag_prompt(prompt: str, chunks: list[dict]) -> str:
    """Format retrieved chunks as RAG context, then the user's question.

    Format: each source block carries a numeric ID + filename + chunk
    index so the LLM can cite them as [1], [2], etc. — chat-ui renders
    those refs as clickable citations using `retrieved_chunks` in the
    response. Numeric IDs (not filenames in the citation marker) keep
    the markup compact and survive even when filenames have spaces or
    special characters.

    Keeps the prompt model-agnostic — the same string works on Llama
    3.1 8B, 3.3 70B, and DeepSeek-R1 70B.
    """
    if not chunks:
        return prompt
    # Each source includes the filename + chunk index so the LLM has
    # enough context to choose between near-duplicate sources from
    # different files. The bracketed [N] is what we ask the LLM to
    # echo back as a citation.
    context = "\n\n".join(
        "[{n}] (source: {src}, chunk {ci})\n{text}".format(
            n=i + 1,
            src=c.get("source", "unknown") or "unknown",
            ci=c.get("chunk_index", 0),
            text=c.get("text", ""),
        )
        for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using the numbered sources below. When a "
        "claim is supported by a source, cite it with [N] matching the "
        "source's number — multiple cites are fine like [1][3]. Do NOT "
        "cite a source that doesn't actually support the claim. If the "
        "sources don't help at all, answer from your own knowledge and "
        "say so explicitly without using any [N] markers.\n\n"
        f"=== Sources ===\n{context}\n\n"
        f"=== Question ===\n{prompt}"
    )


# --- Tools (Phase T) --------------------------------------------------------
#
# The agent loop binds these tools to ChatOpenAI via .bind_tools(). When
# vLLM (started with --enable-auto-tool-choice + --tool-call-parser
# llama3_json) decides to call one, the response carries structured
# tool_calls; node_execute_agent dispatches each to the local handler,
# appends a ToolMessage with the result, and loops back into the model.
#
# Why these four tools (and not more):
#   - calculator: single-shot arithmetic, the canonical "I can do math"
#     demo. numexpr keeps it safe (no eval()).
#   - get_current_time: trivial-but-useful, makes time-aware questions
#     work without retraining the model on current dates.
#   - http_fetch: opens up "what's on this page" questions. Bounded
#     to https + 50 KB to keep the agent from being a side-channel.
#   - search_session_docs: lets the AGENT decide when to RAG-retrieve
#     mid-conversation, vs. the always-pre-retrieve current behavior.
#     Calls rag-service /retrieve with the user's bearer token.
#
# Adding a tool: define a @tool function, append to TOOLS list. The
# bind_tools() in node_execute_agent picks them up at every call.

# Cap on agent loop iterations. 5 is generous — the user's question
# rarely needs more than 2-3 tool calls; 5 is a safety net against
# runaway loops (model keeps calling tools and never producing a
# final text response). After cap, we force a final-answer call
# without tools bound.
AGENT_MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "5"))

# Bound on http_fetch result size. Larger than 50 KB starts straining
# the model's 8K context (roughly 32 KB of text at most).
HTTP_FETCH_MAX_BYTES = 50 * 1024

# Allowed URL schemes for http_fetch. https only — http would be a
# trivial credential exfil channel through the agent. file:// would
# read pod local files. Hard-allowlist.
HTTP_FETCH_ALLOWED_SCHEMES = {"https"}


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic, not for
    word problems — the input must be a pure math expression like
    `2+2*3` or `sqrt(144)/12`. Returns the numeric result as a string.
    Supports +, -, *, /, **, sqrt, sin, cos, log, etc."""
    import numexpr
    try:
        # numexpr.evaluate parses and runs via NumPy without invoking
        # Python eval(). Operators + numeric functions only — no
        # attribute access, no imports.
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"calculator error: {e}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given timezone. timezone
    must be an IANA name like 'America/Los_Angeles', 'Europe/London',
    'Asia/Tokyo', or 'UTC' (default). Returns ISO-8601 format."""
    import pytz
    from datetime import datetime
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.isoformat()
    except pytz.UnknownTimeZoneError:
        return f"unknown timezone '{timezone}'; use IANA names like 'America/Los_Angeles'"


@tool
def http_fetch(url: str) -> str:
    """Fetch a URL and return the response body (truncated to 50 KB).
    Only https URLs are allowed. Use for retrieving content from a
    specific page when the user asks about it. Returns the response
    text, or an error message if the fetch fails."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in HTTP_FETCH_ALLOWED_SCHEMES:
        return f"error: only {HTTP_FETCH_ALLOWED_SCHEMES} URLs allowed; got '{parsed.scheme}'"
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            resp = c.get(url)
            resp.raise_for_status()
        body = resp.text[:HTTP_FETCH_MAX_BYTES]
        if len(resp.text) > HTTP_FETCH_MAX_BYTES:
            body += f"\n\n[truncated to {HTTP_FETCH_MAX_BYTES} bytes; original was {len(resp.text)}]"
        return body
    except httpx.HTTPError as e:
        return f"fetch error: {e}"


@tool
def search_session_docs(query: str) -> str:
    """Search the user's uploaded documents for chunks matching the
    query. Use this when the user references documents they uploaded
    earlier in this chat session, OR when the answer might be in
    their docs. Returns up to 5 matching chunks with their sources.
    Each chunk is numbered [1], [2], etc. — cite them in your answer.

    NOTE: requires session_id and bearer token in the agent's context;
    these are injected by the LangGraph runner — DO NOT pass them as
    arguments."""
    # The actual implementation pulls session_id + auth_token from a
    # contextvar set by node_execute_agent before invoke(). Tools
    # called by LangChain don't see graph state directly — contextvars
    # are the standard pattern for "ambient" args.
    session_id = _AGENT_SESSION_ID.get(None)
    auth_token = _AGENT_AUTH_TOKEN.get(None)
    if not session_id or not auth_token:
        return "search unavailable: no session_id or auth in context"
    try:
        with httpx.Client(timeout=RAG_RETRIEVE_TIMEOUT_SECONDS) as c:
            resp = c.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={"query": query, "session_id": session_id, "top_k": 5},
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            resp.raise_for_status()
        body = resp.json()
        chunks = body.get("chunks", [])
        if not chunks:
            return "no matching chunks in this session"
        return "\n\n".join(
            f"[{i + 1}] (source: {c.get('source', 'unknown')}) {c.get('text', '')[:500]}"
            for i, c in enumerate(chunks)
        )
    except httpx.HTTPError as e:
        return f"search error: {e}"


# Contextvars carry the per-request session_id + auth_token into
# search_session_docs without LangChain's tool decorator seeing them
# as model-visible arguments.
import contextvars
_AGENT_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_AGENT_SESSION_ID", default=None
)
_AGENT_AUTH_TOKEN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_AGENT_AUTH_TOKEN", default=None
)

TOOLS = [calculator, get_current_time, http_fetch, search_session_docs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def node_execute(state: AgentState) -> AgentState:
    """Run the user's prompt against the routed-to variant.

    Two paths:
      A. Tool-capable route (supports_tools=True): agentic loop —
         bind tools, call vLLM, dispatch any tool_calls returned,
         append ToolMessages, loop back. Cap at AGENT_MAX_ITERATIONS.
      B. Plain route: legacy single-shot ChatOpenAI.invoke (no tools
         bound). Used for tiers whose vLLM Deployment doesn't have
         --enable-auto-tool-choice (currently 70B + DeepSeek).

    RAG context (Phase 4 chunks from node_retrieve) is prepended to
    the user prompt either way — tool-capable agents can ALSO call
    search_session_docs mid-conversation if they want fresh chunks
    for a follow-up question.
    """
    cfg = ROUTE_REGISTRY[state["route"]]
    final_prompt = _build_rag_prompt(state["prompt"], state.get("retrieved_chunks", []))
    started = time.monotonic()

    if not cfg.get("supports_tools"):
        # Path B — legacy single-shot. Same code as before tool calling
        # was added; preserves behavior for 70B/DeepSeek tiers.
        client = ChatOpenAI(
            model=cfg["model_name"],
            base_url=cfg["url"],
            api_key="not-required",
            max_tokens=state.get("max_tokens", 512),
            timeout=EXECUTE_TIMEOUT_SECONDS,
        )
        response = client.invoke([HumanMessage(content=final_prompt)])
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return {
            "response": response.content or "",
            "execute_latency_ms": elapsed_ms,
            "tool_iterations": 0,
            "tool_calls_log": [],
        }

    # Path A — agentic loop with tool calling.
    # Bind tools so vLLM gets a tools=[...] payload on every request;
    # Llama 3.1 + llama3_json parser converts the model's tool-call
    # output into structured response.tool_calls. The bound client is
    # the SAME ChatOpenAI but with .bind_tools() applied — LangChain
    # plumbs the schema into the request.
    client = ChatOpenAI(
        model=cfg["model_name"],
        base_url=cfg["url"],
        api_key="not-required",
        max_tokens=state.get("max_tokens", 512),
        timeout=EXECUTE_TIMEOUT_SECONDS,
    ).bind_tools(TOOLS)

    # System prompt — sets the agent persona + reminds the model that
    # tools are available. Without this, Llama 3.1 8B sometimes
    # ignores the tools= field and answers from the chat-template
    # default of "be a helpful assistant".
    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a helpful assistant with access to tools. Use them when "
                "they help: calculator for arithmetic, get_current_time for "
                "time-aware questions, http_fetch when the user references a URL, "
                "and search_session_docs when the user references documents they "
                "uploaded earlier. Otherwise, answer directly from your own "
                "knowledge or the provided context."
            )
        ),
        HumanMessage(content=final_prompt),
    ]

    # Set contextvars so search_session_docs sees the request's
    # session_id + auth_token. Reset in finally so we don't leak
    # one request's token into another request's context.
    sid_token = _AGENT_SESSION_ID.set(state.get("session_id"))
    auth_token_token = _AGENT_AUTH_TOKEN.set(state.get("auth_token"))

    tool_calls_log: list[str] = []
    iterations = 0
    final_response = ""
    try:
        for iterations in range(1, AGENT_MAX_ITERATIONS + 1):
            response = client.invoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # Model produced a plain-text answer — we're done.
                final_response = response.content or ""
                break

            log.info(
                "agent_iter=%d tool_calls=%s",
                iterations,
                [tc.get("name") for tc in tool_calls],
            )

            # Dispatch each tool call. ToolMessage carries the result
            # back into the next loop pass; tool_call_id ties the
            # result to the corresponding model-emitted call (the
            # OpenAI-tools spec requires this matching).
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tool_calls_log.append(name)
                t = TOOLS_BY_NAME.get(name)
                if t is None:
                    result = f"unknown tool '{name}'"
                else:
                    try:
                        result = t.invoke(args)
                    except Exception as e:
                        # Surface tool errors to the model rather than
                        # failing the whole graph — the model can decide
                        # to retry, fall back, or apologize gracefully.
                        result = f"tool {name} failed: {e}"
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc.get("id", ""),
                    )
                )
        else:
            # Loop hit AGENT_MAX_ITERATIONS without a final-text answer.
            # Force one more invoke without tools bound — the model has
            # to commit to a text response now.
            log.warning(
                "agent loop hit max iterations (%d); forcing final answer",
                AGENT_MAX_ITERATIONS,
            )
            no_tools_client = ChatOpenAI(
                model=cfg["model_name"],
                base_url=cfg["url"],
                api_key="not-required",
                max_tokens=state.get("max_tokens", 512),
                timeout=EXECUTE_TIMEOUT_SECONDS,
            )
            final = no_tools_client.invoke(
                messages
                + [
                    HumanMessage(
                        content=(
                            "Stop calling tools and give the final answer to the "
                            "original question using everything you've learned so far."
                        )
                    )
                ]
            )
            final_response = final.content or ""
    finally:
        _AGENT_SESSION_ID.reset(sid_token)
        _AGENT_AUTH_TOKEN.reset(auth_token_token)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "response": final_response,
        "execute_latency_ms": elapsed_ms,
        "tool_iterations": iterations,
        "tool_calls_log": tool_calls_log,
    }


def build_graph() -> StateGraph:
    """Compile the four-node graph. Done once at import time, reused.

    Order: classify → retrieve → ensure_warm → execute → END.
    retrieve sits before ensure_warm so RAG context lookup overlaps
    with cold-start time on cold paths (bge-m3 is always-warm; the
    heavy generator is the one that may need to scale up).
    """
    g: StateGraph = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("ensure_warm", node_ensure_warm)
    g.add_node("execute", node_execute)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "ensure_warm")
    g.add_edge("ensure_warm", "execute")
    g.add_edge("execute", END)
    return g.compile()


GRAPH = build_graph()


# --- Models ----------------------------------------------------------------

class InvokeRequest(BaseModel):
    prompt: str = Field(..., description="The user's natural-language prompt.")
    max_tokens: int = Field(default=512, ge=1, le=8192)
    image_url: Optional[str] = Field(default=None)
    # Phase 4: per-chat-session RAG. When set, the retrieve node calls
    # rag-service /retrieve and prepends matching chunks to the prompt.
    # When unset (or empty), retrieve is a no-op and the original prompt
    # path runs unchanged.
    session_id: Optional[str] = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunkOut(BaseModel):
    """Public-facing chunk shape mirrored from rag-service. Lets chat-ui
    render source citations in the response sidebar without having to
    decode rag-service's response separately."""
    text: str
    source: str
    chunk_index: int
    score: float


class InvokeResponse(BaseModel):
    response: str
    route: str
    cold_start: bool
    warm_wait_seconds: float
    execute_latency_ms: int
    classifier_raw: str
    user: str
    # Phase 4 retrieval telemetry. retrieve_count==0 when session_id
    # wasn't set or rag-service returned no hits — chat-ui can use this
    # to suppress the "Sources" panel.
    retrieve_count: int = 0
    retrieve_ms: int = 0
    retrieved_chunks: list[RetrievedChunkOut] = []
    # langfuse trace id for the v3 SDK's emitted trace, so callers
    # (chat-ui) can deep-link to ${LANGFUSE_HOST}/trace/<id>. Optional
    # because the langfuse callback is None when public/secret keys
    # aren't configured (lab environments without trace export).
    langfuse_trace_id: Optional[str] = None
    # Phase T: agentic loop telemetry. tool_iterations==0 means the
    # routed-to model answered without any tool calls (or the tier
    # doesn't support tools). tool_calls_log lists tool names in
    # call order — chat-ui can render "Tools used: calculator,
    # http_fetch" alongside the response.
    tool_iterations: int = 0
    tool_calls_log: list[str] = []


# --- Routes ----------------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True, "service": "langgraph-service", "version": app.version}


@app.post("/invoke", response_model=InvokeResponse)
def invoke(
    req: InvokeRequest,
    claims: Annotated[dict, Depends(require_jwt)],
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)],
) -> InvokeResponse:
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    log.info(
        "invoke",
        extra={
            "user": user,
            "prompt_len": len(req.prompt),
            "session_id": req.session_id or "",
        },
    )

    # Forward the user's bearer token to the retrieve node so it can
    # call rag-service /retrieve. require_jwt already validated it; we
    # take the same string from the same dependency-injected creds.
    auth_token = creds.credentials if creds else None

    initial: AgentState = {
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "image_url": req.image_url,
        "user": user,
        "session_id": req.session_id,
        "top_k": req.top_k,
        "auth_token": auth_token,
    }
    config: dict = {}
    trace_id: Optional[str] = None
    if _LANGFUSE_CB is not None:
        # v3 SDK takes per-request attribution via metadata — the keys
        # langfuse_user_id / langfuse_tags / langfuse_session_id are
        # consumed by the CallbackHandler when it builds the trace.
        # We also pre-generate the trace_id (32-hex UUID) and pass it
        # via langfuse_trace_id so the caller can deep-link to the
        # Langfuse trace UI without round-tripping the callback's
        # internal state. Generating up-front means the value is known
        # even if the trace export later fails.
        import uuid as _uuid
        trace_id = _uuid.uuid4().hex
        config = {
            "callbacks": [_LANGFUSE_CB],
            "metadata": {
                "langfuse_user_id": user,
                "langfuse_tags": ["langgraph-service"],
                "langfuse_trace_id": trace_id,
            },
        }
    try:
        final_state: AgentState = GRAPH.invoke(initial, config=config)
    except RuntimeError as e:
        # ensure_warm raises RuntimeError on scale failures + ready timeouts.
        # Surface as 502 (upstream service unavailable) rather than 500 so
        # callers can distinguish transient backend issues from logic bugs.
        log.error("graph execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e

    return InvokeResponse(
        response=final_state.get("response", ""),
        route=final_state.get("route", "trivial"),
        cold_start=final_state.get("cold_start", False),
        warm_wait_seconds=final_state.get("warm_wait_seconds", 0.0),
        execute_latency_ms=final_state.get("execute_latency_ms", 0),
        classifier_raw=final_state.get("classifier_raw", ""),
        user=user,
        retrieve_count=final_state.get("retrieve_count", 0),
        retrieve_ms=final_state.get("retrieve_ms", 0),
        retrieved_chunks=[
            RetrievedChunkOut(**c) for c in final_state.get("retrieved_chunks", [])
        ],
        langfuse_trace_id=trace_id,
        tool_iterations=final_state.get("tool_iterations", 0),
        tool_calls_log=final_state.get("tool_calls_log", []),
    )
