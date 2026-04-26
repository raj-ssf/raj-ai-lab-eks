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
from langchain_core.messages import HumanMessage, SystemMessage
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

# JIT scale-up parameters. Karpenter's GPU node typically takes 60-90s to
# come up; vLLM cold-start adds another 2-3 min for 70B AWQ. 5 min total
# is a comfortable upper bound; bump for the 405B path (which can take 6+
# min for S3-Mountpoint weight fault-in).
SCALE_UP_TIMEOUT_SECONDS = 300
SCALE_POLL_INTERVAL_SECONDS = 3

# --- Tool registry: route name → call config -------------------------------

ROUTE_REGISTRY: dict[str, dict] = {
    "trivial": {
        "url": f"{MODEL_TRIVIAL_URL}/v1",
        "model_name": MODEL_TRIVIAL_NAME,
        "deployment": DEPLOY_TRIVIAL,
        "always_on": True,  # Llama 8B is the always-on tier; never JIT-scaled
    },
    "reasoning": {
        "url": f"{MODEL_REASONING_URL}/v1",
        "model_name": MODEL_REASONING_NAME,
        "deployment": DEPLOY_REASONING,
        "always_on": False,
    },
    "hard": {
        "url": f"{MODEL_HARD_URL}/v1",
        "model_name": MODEL_HARD_NAME,
        "deployment": DEPLOY_HARD,
        "always_on": False,
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
    # Set by classify
    route: Literal["trivial", "reasoning", "hard"]
    classifier_raw: str
    # Set by ensure_warm
    cold_start: bool
    warm_wait_seconds: float
    # Set by execute
    response: str
    execute_latency_ms: int


# Classifier system prompt — short, direct, JSON-output. The trick is to
# keep this prompt small enough that the always-on Llama 8B can run it
# fast (~200ms typical) and the routing decision dwarfs end-to-end
# latency only on cold starts.
CLASSIFIER_SYSTEM_PROMPT = """You are a routing classifier. Read the user's prompt and output exactly one word from this set:

- trivial: factual recall, simple math, yes/no questions, very short answers
- reasoning: multi-step thinking, chain-of-thought, math word problems, logic puzzles
- hard: complex tasks (long-form writing, code generation, deep analysis)

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
    # the cheap path handles ambiguous cases.
    route: Literal["trivial", "reasoning", "hard"] = "trivial"
    for candidate in ("hard", "reasoning", "trivial"):
        if candidate in raw:
            route = candidate  # type: ignore[assignment]
            break

    log.info("classified", extra={"route": route, "classifier_raw": raw[:80]})
    return {"route": route, "classifier_raw": raw}


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
        current = _read_replicas(deploy)
    except ApiException as e:
        raise RuntimeError(f"failed to read scale of {LLM_NAMESPACE}/{deploy}: {e.reason}") from e

    if current >= 1:
        # Already warm — no JIT cost. Skip ahead.
        log.info("ensure_warm: variant already warm", extra={"deployment": deploy, "replicas": current})
        return {"cold_start": False, "warm_wait_seconds": 0.0}

    log.info("ensure_warm: scaling up cold variant", extra={"deployment": deploy})
    try:
        _scale_deployment(deploy, replicas=1)
    except ApiException as e:
        raise RuntimeError(f"failed to scale {LLM_NAMESPACE}/{deploy}: {e.reason}") from e

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


def node_execute(state: AgentState) -> AgentState:
    """Run the user's prompt against the routed-to variant."""
    cfg = ROUTE_REGISTRY[state["route"]]
    client = ChatOpenAI(
        model=cfg["model_name"],
        base_url=cfg["url"],
        api_key="not-required",
        max_tokens=state.get("max_tokens", 512),
        timeout=EXECUTE_TIMEOUT_SECONDS,
    )
    started = time.monotonic()
    response = client.invoke([HumanMessage(content=state["prompt"])])
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {"response": response.content or "", "execute_latency_ms": elapsed_ms}


def build_graph() -> StateGraph:
    """Compile the three-node graph. Done once at import time, reused."""
    g: StateGraph = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("ensure_warm", node_ensure_warm)
    g.add_node("execute", node_execute)
    g.add_edge(START, "classify")
    g.add_edge("classify", "ensure_warm")
    g.add_edge("ensure_warm", "execute")
    g.add_edge("execute", END)
    return g.compile()


GRAPH = build_graph()


# --- Models ----------------------------------------------------------------

class InvokeRequest(BaseModel):
    prompt: str = Field(..., description="The user's natural-language prompt.")
    max_tokens: int = Field(default=512, ge=1, le=8192)
    image_url: Optional[str] = Field(default=None)


class InvokeResponse(BaseModel):
    response: str
    route: str
    cold_start: bool
    warm_wait_seconds: float
    execute_latency_ms: int
    classifier_raw: str
    user: str
    # langfuse trace id for the v3 SDK's emitted trace, so callers
    # (chat-ui) can deep-link to ${LANGFUSE_HOST}/trace/<id>. Optional
    # because the langfuse callback is None when public/secret keys
    # aren't configured (lab environments without trace export).
    langfuse_trace_id: Optional[str] = None


# --- Routes ----------------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True, "service": "langgraph-service", "version": app.version}


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest, claims: Annotated[dict, Depends(require_jwt)]) -> InvokeResponse:
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    log.info("invoke", extra={"user": user, "prompt_len": len(req.prompt)})

    initial: AgentState = {
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "image_url": req.image_url,
        "user": user,
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
        langfuse_trace_id=trace_id,
    )
