"""langgraph-service — FastAPI front-end for a LangGraph router agent.

Eleven-node state machine on /invoke (budget + safety bookends + memory
+ query rewriting + reasoning loop):

    START -> budget_check -> safety_input -> load_memory -> rewrite_query
                          -> classify -> retrieve -> ensure_warm
                          -> execute -> reflect -> safety_output
                          -> save_memory -> END
              |                  |                              ^      |
              v                  v                              |______|  (loop, capped)
              END (over budget)  END (refused)

- budget_check: per-user daily request budget enforced via Redis
  INCR + 48h TTL. Cheapest filter, runs first. If over the cap (or
  Redis unreachable + BUDGET_FAIL_MODE=closed), short-circuits to
  END with a budget-exhausted refusal. No-op when BUDGET_ENABLED=false.
- safety_input: Llama Guard 3 8B grades the user's prompt against
  Meta's 14-category hazard taxonomy. If unsafe AND the violated
  categories intersect SAFETY_BLOCK_CATEGORIES, the graph short-
  circuits to END with a refusal response. No-op when
  SAFETY_FILTER_ENABLED=false.
- load_memory: reads recent conversation turns + long-term summary
  from Redis (keyed by user + session_id). No-op when
  MEMORY_ENABLED=false or no session_id. Fail-OPEN on Redis errors.
- rewrite_query: takes the raw prompt + conversation context and
  produces a STANDALONE search query (resolves pronouns, preserves
  technical terms). Sets state.refined_query so node_retrieve uses
  it (same field as Phase T2's reflect node — different writers,
  sequenced correctly). No-op when QUERY_REWRITE_ENABLED=false or
  no conversation context (first turn).
- classify: always-on Llama 3.1 8B answers a JSON-schema classification
  prompt categorizing the user's input as `trivial` / `tuned-lora` /
  `reasoning` / `hard`.
- retrieve: per-session RAG retrieval against rag-service. On loop
  re-entry, uses `refined_query` (set by reflect) instead of the
  original prompt and accumulates chunks (deduped) across cycles.
- ensure_warm: if the routed-to variant is at replicas=0, scale to 1 and
  wait for Ready.
- execute: HTTP POST to the chosen variant's vLLM OpenAI-compat endpoint.
  For tool-capable tiers, runs an agentic tool-call loop bounded by
  AGENT_MAX_ITERATIONS (per-execute, distinct from the graph-level cap).
- reflect: gates whether to loop. Asks the cheap Llama 8B to decide if
  another retrieval cycle (with a new query) would meaningfully improve
  the draft answer. If yes AND we haven't hit MAX_REASONING_CYCLES,
  routes back to retrieve; otherwise routes to safety_output.
- safety_output: Llama Guard 3 8B grades the model's draft. If unsafe,
  replaces `response` with the refusal message before END.
- save_memory: appends the (prompt, response) turn to Redis memory.
  Skips on safety-blocked or budget-blocked paths. Re-summarizes
  every MEMORY_SUMMARIZE_AFTER_TURNS turns to bound recent-turns
  list size.

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
        # Tools intentionally DISABLED for the LoRA-merged tier.
        # Alpaca LoRA was trained on instruction-following text only;
        # it never saw Llama 3.1's native tool-call grammar
        # (<|python_tag|>{...}<|eom_id|>). At inference, the LoRA-
        # shifted distribution emits tool calls as plain JSON in the
        # text response, which vLLM's llama3_json parser doesn't
        # detect — calls go through as text, tool_calls_log is empty,
        # and the user sees raw JSON in chat-ui. Verified empirically
        # 2026-04-29 in the Phase T smoke. Re-enable only after a
        # fine-tune that includes tool-call data (e.g., on the
        # toolbench dataset).
        "supports_tools": False,
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
    # Phase T2: graph-level reasoning loop bookkeeping. cycles counts
    # full retrieve→execute→reflect passes (capped at MAX_REASONING_CYCLES).
    # refined_query, when set by node_reflect, is consumed by node_retrieve
    # as the search string for the next pass — replaces the original prompt
    # for retrieval ONLY (the user's prompt itself stays the question the
    # final answer addresses). needs_more_context drives the conditional
    # edge after reflect. reflection_log keeps a per-cycle audit trail
    # surfaced in the API response for debugging + Langfuse correlation.
    cycles: int
    refined_query: Optional[str]
    needs_more_context: bool
    reflection_log: list[str]
    # Phase #4: content safety bookkeeping.
    #   safety_input_verdict / safety_output_verdict: "safe", "unsafe",
    #     "skipped" (filter disabled), or "fail_open"/"fail_closed"
    #     (Llama Guard unreachable, fell back to configured fail mode).
    #   safety_categories: list of S-codes that triggered the block,
    #     in the order Llama Guard returned them. Empty for "safe" or
    #     "skipped" verdicts.
    #   safety_action: terminal disposition — "passed", "blocked_input",
    #     "blocked_output", "disabled". Drives the conditional edges
    #     after each safety node.
    #   safety_input_ms / safety_output_ms: per-node latency for
    #     telemetry budgeting (Llama Guard call time on a warm pod
    #     is ~150-300 ms; cold-start adds the usual 8 min).
    safety_input_verdict: Literal["safe", "unsafe", "skipped", "fail_open", "fail_closed"]
    safety_output_verdict: Literal["safe", "unsafe", "skipped", "fail_open", "fail_closed"]
    safety_categories: list[str]
    safety_action: Literal["passed", "blocked_input", "blocked_output", "disabled"]
    safety_input_ms: int
    safety_output_ms: int
    # Phase #5: cost-guardrail bookkeeping.
    #   budget_action: terminal disposition — "passed" | "exceeded" |
    #     "disabled" | "fail_open" | "fail_closed". Drives the conditional
    #     edge after node_budget_check.
    #   budget_consumed: post-INCR total for the user this UTC day.
    #     Sourced from Redis. 0 when filter disabled.
    #   budget_remaining: BUDGET_REQUESTS_PER_DAY - budget_consumed,
    #     surfaced so chat-ui can render a "X requests left today"
    #     widget. Negative if the user just exceeded.
    budget_action: Literal["passed", "exceeded", "disabled", "fail_open", "fail_closed"]
    budget_consumed: int
    budget_remaining: int
    # Phase #6: conversational memory + query rewriting bookkeeping.
    #   memory_summary: long-term summary of the conversation (3-4
    #     sentences, regenerated every MEMORY_SUMMARIZE_AFTER_TURNS turns).
    #   memory_recent_turns: last MEMORY_RECENT_TURNS turn dicts
    #     ({prompt, response, ts}). Populated by node_load_memory; used
    #     by node_rewrite_query for context resolution.
    #   query_rewritten: True if rewrite ran AND produced a refined_query
    #     different from the original prompt. False on disabled / no
    #     conversation context / rewrite returning identity.
    #   original_prompt: preserved across rewrite so /invoke can return
    #     both the verbatim user input AND the standalone search query.
    #   memory_load_ms / memory_save_ms / query_rewrite_ms: per-node
    #     latencies for telemetry.
    memory_summary: str
    memory_recent_turns: list[dict]
    query_rewritten: bool
    original_prompt: str
    memory_load_ms: int
    memory_save_ms: int
    query_rewrite_ms: int


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


# --- Content safety (Phase #4) ---------------------------------------------
#
# Llama Guard 3 8B is a Llama-family classifier fine-tuned on Meta's
# 14-category hazard taxonomy. Its tokenizer ships a chat template that
# wraps an input conversation in a safety-categories prompt and asks
# the model to output 'safe' or 'unsafe\n<S-codes>'. vLLM applies the
# template automatically when we POST /v1/chat/completions with
# model=llama-guard-3-8b, so the call site is the same as any other
# OpenAI-compat chat call.
#
# Two graph nodes use this:
#   safety_input  — runs BEFORE classify. Saves cost (a malicious
#                   prompt never pays for retrieve+execute), AND
#                   shifts content blocks earlier than late-binding
#                   would (a user typing an unsafe prompt sees the
#                   refusal in <300ms instead of 5+ sec).
#   safety_output — runs AFTER reflect. Catches the cases where the
#                   user's prompt was benign but the model produced
#                   bad output — this happens with adversarial
#                   prompt-injection attacks and with rare model
#                   misalignment.
#
# Both nodes share the helper below. The role= argument controls
# which speaker's content Llama Guard scores (it grades different
# axes for user vs assistant turns).


def _llama_guard_check(
    role: Literal["user", "assistant"],
    content: str,
    user_context: Optional[str] = None,
) -> tuple[str, list[str], int]:
    """Score `content` for safety via vllm-llama-guard-3-8b.

    Returns (verdict, categories, latency_ms):
      verdict     "safe" | "unsafe" | "fail_open" | "fail_closed"
      categories  list of S-codes that triggered the unsafe verdict
                  (e.g. ["S1", "S10"]); empty otherwise
      latency_ms  wall time for the round-trip

    user_context: when scoring an assistant turn (role="assistant"),
      Llama Guard's chat template requires a USER turn before the
      ASSISTANT turn — its safety prompt is "score the LAST agent
      message in the conversation," and a conversation that starts
      with the assistant has no "last agent" since there's no
      preceding context. Sending only an AIMessage returns vLLM 500
      (chat-template assertion failure). Pass the original user prompt
      as user_context so the model sees the real exchange. Caller
      passes state["prompt"] from node_safety_output.

    Failure mode (Llama Guard unreachable, malformed output, etc.):
      SAFETY_FAIL_MODE=open   -> verdict="fail_open", treated as safe
      SAFETY_FAIL_MODE=closed -> verdict="fail_closed", treated as unsafe
    Either way the verdict string surfaces in telemetry so the operator
    can audit how often the filter is actually firing vs degrading.
    """
    started = time.monotonic()

    # Llama Guard's chat template auto-wraps the conversation in the
    # safety-categories prompt. We just pass the raw user/assistant
    # turn we want graded; vLLM + the model's tokenizer config handle
    # the rest. Use the standard OpenAI-compat API for portability.
    client = ChatOpenAI(
        model=SAFETY_LLAMA_GUARD_MODEL,
        base_url=SAFETY_LLAMA_GUARD_URL,
        api_key="not-required",
        temperature=0.0,
        # Output is "safe" (4 tokens) or "unsafe\nS1,S2,..." (~30 tokens).
        # 96 is generous; bumping wastes time on a guarantee we already
        # have from the model's training.
        max_tokens=96,
        timeout=SAFETY_TIMEOUT_SECONDS,
    )
    if role == "user":
        msgs: list = [HumanMessage(content=content)]
    else:
        # Build a user→assistant conversation for the chat template.
        # Use the actual prior user prompt if the caller supplied it;
        # otherwise a generic placeholder still keeps the template
        # well-formed (Llama Guard scores the LAST agent turn against
        # the categories regardless of how informative the user turn
        # was).
        msgs = [
            HumanMessage(content=user_context or "(prior user message not provided)"),
            AIMessage(content=content),
        ]
    try:
        response = client.invoke(msgs)
        raw = (response.content or "").strip()
    except Exception as e:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.warning("llama_guard call failed: %s (mode=%s)", e, SAFETY_FAIL_MODE)
        verdict = "fail_open" if SAFETY_FAIL_MODE == "open" else "fail_closed"
        return (verdict, [], elapsed_ms)

    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Parse: first line is "safe" | "unsafe". Optional second line is
    # a comma-separated S-code list. Be forgiving on whitespace/case.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        log.warning("llama_guard returned empty body; treating per fail mode")
        return ("fail_open" if SAFETY_FAIL_MODE == "open" else "fail_closed", [], elapsed_ms)
    head = lines[0].lower()
    if head == "safe":
        return ("safe", [], elapsed_ms)
    if head == "unsafe":
        cats: list[str] = []
        if len(lines) > 1:
            cats = [
                c.strip().upper()
                for c in lines[1].split(",")
                if c.strip()
            ]
        return ("unsafe", cats, elapsed_ms)
    # Anything else means the model misbehaved (chat template
    # mismatch, fine-tune drift, etc.). Treat as fail mode.
    log.warning("llama_guard unparseable head=%r; treating per fail mode", head[:32])
    return ("fail_open" if SAFETY_FAIL_MODE == "open" else "fail_closed", [], elapsed_ms)


def _is_blocking(verdict: str, categories: list[str]) -> bool:
    """Decide whether a verdict + category list should terminate the request.

    "safe" / "skipped" / "fail_open" -> pass through.
    "fail_closed"                    -> block (operator chose this posture).
    "unsafe"                         -> block IFF any category is in the
                                        configured block-list. If the
                                        block-list is empty, any unsafe
                                        verdict blocks regardless of
                                        category — useful for "deny
                                        everything Llama Guard flags".
    """
    if verdict == "fail_closed":
        return True
    if verdict != "unsafe":
        return False
    if not SAFETY_BLOCK_CATEGORIES:
        return True  # block anything unsafe
    return any(c in SAFETY_BLOCK_CATEGORIES for c in categories)


def node_budget_check(state: AgentState) -> AgentState:
    """First-line cost guardrail. Increment Redis counter, gate on cap.

    Redis pattern:
      INCR cost:<user>:<YYYY-MM-DD> -> returns post-increment value
      EXPIRE cost:<user>:<YYYY-MM-DD> 172800   (48h, idempotent)
    The EXPIRE is set every call (cheap; a no-op if already set with
    a longer TTL) to handle the edge where a key was created before
    EXPIRE wiring landed and never got a TTL.

    Returns budget_action of:
      "disabled"     -> filter off (BUDGET_ENABLED false or limit 0)
      "passed"       -> within budget; pass through
      "exceeded"     -> over the cap; pre-populate response with
                        BUDGET_REFUSAL_MESSAGE and short-circuit to END
      "fail_open"    -> Redis unreachable, BUDGET_FAIL_MODE=open;
                        treated as passed for routing
      "fail_closed"  -> Redis unreachable, BUDGET_FAIL_MODE=closed;
                        treated as exceeded for routing
    """
    if not BUDGET_ENABLED or BUDGET_REQUESTS_PER_DAY <= 0:
        return {
            "budget_action": "disabled",
            "budget_consumed": 0,
            "budget_remaining": 0,
        }

    user = state.get("user", "unknown")
    redis_client = _get_redis()
    if redis_client is None:
        # Init failed earlier; honor fail mode.
        action: Literal["fail_open", "fail_closed"] = (
            "fail_open" if BUDGET_FAIL_MODE == "open" else "fail_closed"
        )
        log.warning("budget_check: redis client unavailable, action=%s", action)
        if action == "fail_closed":
            return {
                "budget_action": "fail_closed",
                "budget_consumed": -1,
                "budget_remaining": 0,
                "response": BUDGET_REFUSAL_MESSAGE,
                "route": "trivial",
                "classifier_raw": "(budget-fail-closed)",
            }
        return {
            "budget_action": "fail_open",
            "budget_consumed": -1,
            "budget_remaining": BUDGET_REQUESTS_PER_DAY,
        }

    key = _budget_today_key(user)
    try:
        # Pipelined INCR + EXPIRE keeps the rate-limit hot path to a
        # single round-trip. EXPIRE returns 1 on success, 0 if key
        # didn't exist (impossible right after INCR) — we ignore the
        # value either way.
        pipe = redis_client.pipeline(transaction=False)
        pipe.incr(key, amount=1)
        pipe.expire(key, 60 * 60 * 48)  # 48h
        results = pipe.execute()
        consumed = int(results[0])
    except Exception as e:
        action = "fail_open" if BUDGET_FAIL_MODE == "open" else "fail_closed"
        log.warning("budget_check: redis op failed: %s, action=%s", e, action)
        if action == "fail_closed":
            return {
                "budget_action": "fail_closed",
                "budget_consumed": -1,
                "budget_remaining": 0,
                "response": BUDGET_REFUSAL_MESSAGE,
                "route": "trivial",
                "classifier_raw": "(budget-fail-closed)",
            }
        return {
            "budget_action": "fail_open",
            "budget_consumed": -1,
            "budget_remaining": BUDGET_REQUESTS_PER_DAY,
        }

    remaining = BUDGET_REQUESTS_PER_DAY - consumed
    log.info(
        "budget_check user=%s consumed=%d remaining=%d limit=%d",
        user,
        consumed,
        remaining,
        BUDGET_REQUESTS_PER_DAY,
    )

    if consumed > BUDGET_REQUESTS_PER_DAY:
        # Over the cap. Pre-populate response and skip to END via the
        # conditional edge below. Note the strict > (not >=) — the
        # Nth request is the LAST allowed one, the (N+1)th is over.
        return {
            "budget_action": "exceeded",
            "budget_consumed": consumed,
            "budget_remaining": remaining,
            "response": BUDGET_REFUSAL_MESSAGE,
            "route": "trivial",
            "classifier_raw": "(budget-exceeded)",
        }

    return {
        "budget_action": "passed",
        "budget_consumed": consumed,
        "budget_remaining": remaining,
    }


def _route_after_budget_check(state: AgentState) -> Literal["safety_input", "__end__"]:
    """Conditional edge: if budget exceeded or fail_closed, terminate.

    Defense-in-depth pattern matches _route_after_safety_input — even
    if a future edit produces an "exceeded" action without setting
    response, this edge still routes correctly to END.
    """
    if state.get("budget_action") in ("exceeded", "fail_closed"):
        return END
    return "safety_input"


def node_safety_input(state: AgentState) -> AgentState:
    """Score the user's prompt; block early if Llama Guard flags it."""
    if not SAFETY_FILTER_ENABLED:
        return {
            "safety_input_verdict": "skipped",
            "safety_categories": [],
            "safety_action": "passed",
            "safety_input_ms": 0,
        }

    verdict, categories, ms = _llama_guard_check("user", state["prompt"])
    log.info(
        "safety_input verdict=%s categories=%s ms=%d",
        verdict,
        categories,
        ms,
    )

    if _is_blocking(verdict, categories):
        # Pre-populate the response so the conditional edge can route
        # straight to END without execute ever firing. The downstream
        # InvokeResponse converts this into the user-facing refusal.
        return {
            "safety_input_verdict": verdict,
            "safety_categories": categories,
            "safety_action": "blocked_input",
            "safety_input_ms": ms,
            "response": SAFETY_REFUSAL_MESSAGE,
            # Skip-route bookkeeping so chat-ui can render a clean
            # "blocked" badge: classifier didn't run, retrieve didn't
            # run, nothing else has meaningful data here.
            "route": "trivial",
            "classifier_raw": "(safety-blocked)",
        }

    return {
        "safety_input_verdict": verdict,
        "safety_categories": categories,
        "safety_action": "passed",
        "safety_input_ms": ms,
    }


def node_safety_output(state: AgentState) -> AgentState:
    """Score the model's draft answer; replace with refusal if unsafe."""
    if not SAFETY_FILTER_ENABLED:
        return {
            "safety_output_verdict": "skipped",
            "safety_output_ms": 0,
        }

    # If the input was already blocked, the response is already the
    # refusal message — re-checking it is wasteful and would fail-open
    # tautologically. Short-circuit.
    if state.get("safety_action") == "blocked_input":
        return {
            "safety_output_verdict": "skipped",
            "safety_output_ms": 0,
        }

    response = state.get("response", "") or ""
    if not response:
        # Empty response — nothing to score. Pass through; let the
        # client deal with the empty body.
        return {
            "safety_output_verdict": "skipped",
            "safety_output_ms": 0,
        }

    # Pass the original prompt as user_context so Llama Guard's chat
    # template sees a well-formed user→assistant conversation. Without
    # this, vLLM 500s on the lone-assistant turn (caught in the Phase
    # #4 activation smoke 2026-04-29).
    verdict, categories, ms = _llama_guard_check(
        "assistant", response, user_context=state.get("prompt"),
    )
    log.info(
        "safety_output verdict=%s categories=%s ms=%d",
        verdict,
        categories,
        ms,
    )

    if _is_blocking(verdict, categories):
        # Preserve the existing safety_categories from the input check
        # if they're more specific; otherwise overwrite. Output-block
        # categories are typically a superset (model produced new
        # hazard codes the input didn't have).
        merged_cats = list(
            dict.fromkeys((state.get("safety_categories") or []) + categories)
        )
        return {
            "safety_output_verdict": verdict,
            "safety_categories": merged_cats,
            "safety_action": "blocked_output",
            "safety_output_ms": ms,
            "response": SAFETY_REFUSAL_MESSAGE,
        }

    return {
        "safety_output_verdict": verdict,
        "safety_output_ms": ms,
    }


def _route_after_safety_input(state: AgentState) -> Literal["classify", "__end__"]:
    """Conditional edge: if input was blocked, terminate; else continue.

    Same defense-in-depth pattern as _route_after_reflect — the cap is
    re-checked here even though node_safety_input already pre-populated
    state appropriately. If a future change to node_safety_input ever
    produces a state with safety_action="blocked_input" but doesn't set
    response, this edge still routes correctly.
    """
    if state.get("safety_action") == "blocked_input":
        return END
    return "classify"


# --- Memory + query rewriting (Phase #6) -----------------------------------
#
# The three memory nodes share one Redis client (the same _get_redis()
# helper Phase #5 set up). They store conversation state under two keys
# per (user, session_id):
#
#   mem:<user>:<session>:turns    Redis LIST of JSON turn dicts.
#                                 LPUSH on save, LRANGE on load. Capped
#                                 at MEMORY_RECENT_TURNS via LTRIM.
#   mem:<user>:<session>:summary  Redis STRING with the long-term summary.
#                                 Updated every MEMORY_SUMMARIZE_AFTER_TURNS
#                                 turns by a Llama 8B summarization call.
#
# Both keys carry MEMORY_TTL_SECONDS (default 7d). Each load/save
# refreshes the TTL — active conversations stay alive, idle sessions
# self-clean.
#
# Failure modes (Redis unreachable, malformed data, etc.) all fail-OPEN
# — memory degrades gracefully to "no context loaded" and the request
# proceeds with raw prompt. Memory is enrichment, not the critical path.


_QUERY_REWRITE_SYSTEM_PROMPT = """You are a search query rewriter for a chat assistant with RAG retrieval.

Read the conversation history and the user's latest message, then output a STANDALONE search query that captures the user's intent without requiring conversation context.

Rules:
- If the latest message is already a clear standalone question, output it unchanged.
- Replace pronouns ("it", "that", "this", "they") with what they refer to from the conversation.
- Preserve specific terms verbatim: function names, file names, technical jargon, model names.
- Output ONLY the rewritten query as a single line. No explanation, no quotes, no prefix."""


_MEMORY_SUMMARIZE_SYSTEM_PROMPT = """You are a conversation summarizer. Produce a 3-4 sentence summary of the conversation below capturing:
1. What the user is overall trying to accomplish.
2. Key facts, names, or decisions established.
3. Any preferences expressed (e.g. "user prefers concise answers", "user works in Python").

Keep it terse — this is conversation memory, not a transcript. Output ONLY the summary text. No headers, no bullet list."""


def node_load_memory(state: AgentState) -> AgentState:
    """Read recent turns + summary for (user, session_id) from Redis."""
    if not MEMORY_ENABLED:
        return {
            "memory_summary": "",
            "memory_recent_turns": [],
            "memory_load_ms": 0,
        }
    user = state.get("user", "unknown")
    session_id = state.get("session_id")
    if not session_id:
        return {
            "memory_summary": "",
            "memory_recent_turns": [],
            "memory_load_ms": 0,
        }

    started = time.monotonic()
    redis_client = _get_redis()
    if redis_client is None:
        log.warning("load_memory: redis unavailable, fail-open")
        return {
            "memory_summary": "",
            "memory_recent_turns": [],
            "memory_load_ms": int((time.monotonic() - started) * 1000),
        }

    try:
        # Pipelined LRANGE + GET in one round-trip
        pipe = redis_client.pipeline(transaction=False)
        pipe.lrange(_memory_turns_key(user, session_id), 0, MEMORY_RECENT_TURNS - 1)
        pipe.get(_memory_summary_key(user, session_id))
        results = pipe.execute()
        raw_turns: list[str] = results[0] or []
        summary: str = results[1] or ""
    except Exception as e:
        log.warning("load_memory: redis op failed: %s, fail-open", e)
        return {
            "memory_summary": "",
            "memory_recent_turns": [],
            "memory_load_ms": int((time.monotonic() - started) * 1000),
        }

    import json as _json
    parsed_turns: list[dict] = []
    for entry in raw_turns:
        try:
            parsed_turns.append(_json.loads(entry))
        except (_json.JSONDecodeError, TypeError):
            # Defensive — skip a corrupted entry rather than fail the
            # whole load. Could happen if memory format ever changes.
            continue

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "load_memory user=%s session=%s turns=%d summary_chars=%d ms=%d",
        user, session_id, len(parsed_turns), len(summary), elapsed_ms,
    )
    return {
        "memory_summary": summary,
        "memory_recent_turns": parsed_turns,
        "memory_load_ms": elapsed_ms,
    }


def node_rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the user prompt as a standalone search query.

    Uses Llama 3.1 8B (the always-on classifier tier) at temp=0.
    Sets state.refined_query so node_retrieve picks it up (the same
    field Phase T2's reflect node uses on loop iterations — different
    writers, sequenced correctly by graph topology).

    Skips when:
      - QUERY_REWRITE_ENABLED is false
      - No conversation context (no recent turns AND no summary) — the
        prompt IS already standalone, no rewrite would change it
      - LLM call fails (fail-open, retrieve uses original prompt)
    """
    original_prompt = state.get("prompt", "")
    if not QUERY_REWRITE_ENABLED:
        return {
            "original_prompt": original_prompt,
            "query_rewritten": False,
            "query_rewrite_ms": 0,
        }

    recent_turns = state.get("memory_recent_turns") or []
    summary = state.get("memory_summary") or ""
    if not recent_turns and not summary:
        # First turn of a conversation — prompt IS the standalone query.
        return {
            "original_prompt": original_prompt,
            "query_rewritten": False,
            "query_rewrite_ms": 0,
        }

    # Build the user-facing prompt: summary (if any) + last few turns
    # + the new message. Keep the recent turns short — only enough for
    # pronoun resolution, not full transcript.
    parts: list[str] = []
    if summary:
        parts.append(f"Conversation so far (summary): {summary}")
    if recent_turns:
        parts.append("Recent turns:")
        # Memory list is newest-first (LPUSH); reverse for chronological
        # order in the prompt.
        for turn in reversed(recent_turns[-MEMORY_RECENT_TURNS:]):
            user_msg = (turn.get("prompt") or "")[:200]
            ai_msg = (turn.get("response") or "")[:200]
            parts.append(f"  User: {user_msg}")
            parts.append(f"  Assistant: {ai_msg}")
    parts.append(f"\nLatest message: {original_prompt}")
    parts.append("\nStandalone search query:")
    user_msg_text = "\n".join(parts)

    started = time.monotonic()
    cfg_trivial = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg_trivial["model_name"],
        base_url=cfg_trivial["url"],
        api_key="not-required",
        temperature=0.0,
        max_tokens=QUERY_REWRITE_MAX_TOKENS,
        timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
    )
    try:
        response = client.invoke([
            SystemMessage(content=_QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg_text),
        ])
        rewritten = (response.content or "").strip()
    except Exception as e:
        log.warning("rewrite_query: LLM call failed: %s, fail-open", e)
        return {
            "original_prompt": original_prompt,
            "query_rewritten": False,
            "query_rewrite_ms": int((time.monotonic() - started) * 1000),
        }

    elapsed_ms = int((time.monotonic() - started) * 1000)

    # If the rewrite is empty or identical to the original, skip setting
    # refined_query — no point in retrieve doing extra work for the same
    # search string.
    if not rewritten or rewritten == original_prompt.strip():
        log.info("rewrite_query: identity (no rewrite needed) ms=%d", elapsed_ms)
        return {
            "original_prompt": original_prompt,
            "query_rewritten": False,
            "query_rewrite_ms": elapsed_ms,
        }

    log.info(
        "rewrite_query rewrote=%r -> %r ms=%d",
        original_prompt[:80], rewritten[:80], elapsed_ms,
    )
    return {
        "original_prompt": original_prompt,
        "query_rewritten": True,
        "refined_query": rewritten,  # consumed by node_retrieve
        "query_rewrite_ms": elapsed_ms,
    }


def node_save_memory(state: AgentState) -> AgentState:
    """Append the new turn at the end of the graph.

    Runs AFTER safety_output so we save what the user actually saw
    (e.g. refusal text on safety block, not the unsafe draft). Skips
    on safety-blocked paths AND budget-blocked paths so we don't
    pollute the conversation history with failed attempts.

    If the post-save turn count crosses MEMORY_SUMMARIZE_AFTER_TURNS,
    re-summarize and update the summary key.
    """
    if not MEMORY_ENABLED:
        return {"memory_save_ms": 0}
    user = state.get("user", "unknown")
    session_id = state.get("session_id")
    if not session_id:
        return {"memory_save_ms": 0}

    # Don't save blocked attempts
    if state.get("safety_action") in ("blocked_input", "blocked_output"):
        return {"memory_save_ms": 0}
    if state.get("budget_action") in ("exceeded", "fail_closed"):
        return {"memory_save_ms": 0}

    started = time.monotonic()
    redis_client = _get_redis()
    if redis_client is None:
        log.warning("save_memory: redis unavailable, skipping")
        return {"memory_save_ms": int((time.monotonic() - started) * 1000)}

    import json as _json
    turn = {
        "prompt": state.get("original_prompt") or state.get("prompt") or "",
        "response": state.get("response") or "",
        "ts": time.time(),
    }
    turns_key = _memory_turns_key(user, session_id)
    summary_key = _memory_summary_key(user, session_id)

    try:
        # LPUSH new turn at head, LTRIM to bounded length, set TTL.
        # Single round-trip via pipeline.
        pipe = redis_client.pipeline(transaction=False)
        pipe.lpush(turns_key, _json.dumps(turn))
        # Keep up to 2x MEMORY_RECENT_TURNS so summarization has more
        # input than rewrite uses. Older entries get trimmed.
        pipe.ltrim(turns_key, 0, MEMORY_RECENT_TURNS * 2 - 1)
        pipe.expire(turns_key, MEMORY_TTL_SECONDS)
        pipe.expire(summary_key, MEMORY_TTL_SECONDS)
        pipe.llen(turns_key)
        results = pipe.execute()
        new_len = int(results[-1])
    except Exception as e:
        log.warning("save_memory: redis op failed: %s", e)
        return {"memory_save_ms": int((time.monotonic() - started) * 1000)}

    # Re-summarize every Nth turn. Cheap LLM call (Llama 8B, ~200 tokens
    # in / 80 tokens out), runs in the request's tail latency. If you
    # want it off the hot path, move to a background goroutine — the
    # pattern would be: fire+forget a separate thread or async task that
    # writes the summary key. For lab simplicity, inline.
    if new_len > 0 and new_len % MEMORY_SUMMARIZE_AFTER_TURNS == 0:
        _summarize_memory(redis_client, user, session_id, turns_key, summary_key)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "save_memory user=%s session=%s turn_count=%d ms=%d",
        user, session_id, new_len, elapsed_ms,
    )
    return {"memory_save_ms": elapsed_ms}


def _summarize_memory(
    redis_client,
    user: str,
    session_id: str,
    turns_key: str,
    summary_key: str,
) -> None:
    """Regenerate the long-term summary from all recent turns.

    Best-effort — failures log + no-op (the next save will retry on
    the same threshold). The previous summary stays valid until
    overwritten.
    """
    import json as _json
    try:
        raw_turns = redis_client.lrange(turns_key, 0, -1) or []
    except Exception as e:
        log.warning("summarize_memory: lrange failed: %s", e)
        return

    parts: list[str] = []
    for raw in reversed(raw_turns):  # chronological
        try:
            t = _json.loads(raw)
            parts.append(f"User: {(t.get('prompt') or '')[:300]}")
            parts.append(f"Assistant: {(t.get('response') or '')[:300]}")
        except (_json.JSONDecodeError, TypeError):
            continue
    if not parts:
        return

    cfg_trivial = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg_trivial["model_name"],
        base_url=cfg_trivial["url"],
        api_key="not-required",
        temperature=0.0,
        max_tokens=192,  # 3-4 sentences fits comfortably
        timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
    )
    try:
        response = client.invoke([
            SystemMessage(content=_MEMORY_SUMMARIZE_SYSTEM_PROMPT),
            HumanMessage(content="Conversation:\n" + "\n".join(parts)),
        ])
        new_summary = (response.content or "").strip()
    except Exception as e:
        log.warning("summarize_memory: LLM call failed: %s", e)
        return

    if not new_summary:
        return
    try:
        redis_client.set(summary_key, new_summary, ex=MEMORY_TTL_SECONDS)
        log.info(
            "summarize_memory user=%s session=%s summary_chars=%d",
            user, session_id, len(new_summary),
        )
    except Exception as e:
        log.warning("summarize_memory: set failed: %s", e)


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
    """Phase 4 + T2: per-session RAG retrieval, cycle-aware.

    Calls rag-service POST /retrieve with the user's prompt + session_id.
    rag-service embeds via vllm-bge-m3 and queries the `documents`
    Qdrant collection filtered by (session_id AND user-from-JWT). The
    returned chunks land in state and are prepended to the prompt by
    node_execute as RAG context.

    Cycle-aware behavior (T2):
      - First pass: query = state["prompt"], chunks list starts empty.
      - Re-entry from reflect node: query = state["refined_query"]
        (set by node_reflect's decision JSON). New chunks are deduped
        against the chunks already in state and APPENDED — context
        accumulates across cycles rather than getting overwritten.
        Dedupe key is (source, chunk_index): the same Qdrant point
        returned by two queries should not be re-injected.
      - refined_query is reset to None on the way out so the NEXT
        reflect call starts fresh.

    Skips when:
      - session_id is None or empty (no chat session — chat-ui upload
        flow hasn't run; nothing to retrieve)
      - rag-service is unreachable (logged + treated as zero hits; we
        don't want a flaky retrieve to break /invoke entirely)
    """
    session_id = state.get("session_id")
    if not session_id:
        return {
            "retrieved_chunks": state.get("retrieved_chunks", []) or [],
            "retrieve_count": state.get("retrieve_count", 0),
            "retrieve_ms": 0,
            "refined_query": None,
        }

    auth_token = state.get("auth_token")
    if not auth_token:
        # Should never happen — /invoke handler always sets it when
        # session_id is set. Defensive log + skip rather than 500.
        log.warning("retrieve: session_id set but auth_token missing; skipping")
        return {
            "retrieved_chunks": state.get("retrieved_chunks", []) or [],
            "retrieve_count": state.get("retrieve_count", 0),
            "retrieve_ms": 0,
            "refined_query": None,
        }

    # T2: prefer the refined query the reflect node emitted on the previous
    # cycle. On the first pass, refined_query is None and we use the
    # original prompt. After this consumes it, we reset it to None so the
    # next reflect call doesn't see stale data.
    query_for_retrieval = state.get("refined_query") or state["prompt"]
    existing_chunks = list(state.get("retrieved_chunks") or [])
    existing_keys = {
        (c.get("source", ""), c.get("chunk_index", 0)) for c in existing_chunks
    }

    started = time.monotonic()
    new_chunks: list[dict] = []
    try:
        with httpx.Client(timeout=RAG_RETRIEVE_TIMEOUT_SECONDS) as client:
            resp = client.post(
                f"{RAG_SERVICE_URL}/retrieve",
                json={
                    "query": query_for_retrieval,
                    "session_id": session_id,
                    "top_k": state.get("top_k", RAG_TOP_K_DEFAULT),
                },
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            resp.raise_for_status()
            body = resp.json()
            fetched = body.get("chunks", []) or []
            # Dedupe: skip any chunk we already have. Same Qdrant point
            # would resurface if the refined query is similar to the
            # original; injecting it twice wastes context budget.
            for c in fetched:
                key = (c.get("source", ""), c.get("chunk_index", 0))
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                new_chunks.append(c)
    except httpx.HTTPError as e:
        # Don't fail the request — RAG is enrichment, not the critical
        # path. Log + return empty so execute proceeds without context.
        log.warning("retrieve failed (continuing without context): %s", e)

    combined = existing_chunks + new_chunks
    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "retrieve",
        extra={
            "session_id": session_id,
            "query": query_for_retrieval[:80],
            "new_chunks": len(new_chunks),
            "total_chunks": len(combined),
            "retrieve_ms": elapsed_ms,
        },
    )
    return {
        "retrieved_chunks": combined,
        "retrieve_count": len(combined),
        "retrieve_ms": elapsed_ms,
        "refined_query": None,
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

# Phase T2: graph-level reasoning loop cap. Distinct from
# AGENT_MAX_ITERATIONS — that one bounds the per-execute tool-call
# loop inside ONE LLM exchange. This one bounds the count of full
# retrieve→execute→reflect cycles. The reflect node reads the draft
# answer and decides whether a NEW retrieval (with a refined query)
# would improve things; if so, the graph loops back to retrieve.
# 3 is enough to model "draft → realize gap → re-retrieve → final"
# without thrashing. Cap exists to bound worst-case latency + cost.
MAX_REASONING_CYCLES = int(os.environ.get("MAX_REASONING_CYCLES", "3"))
REFLECT_MAX_TOKENS = 96
REFLECT_TEMPERATURE = 0.0
REFLECT_TIMEOUT_SECONDS = 15

# --- Phase #4: content safety (Llama Guard 3 8B) ---------------------------
#
# Two filter points in the graph: safety_input runs BEFORE classify
# (cheap path: a malicious prompt never pays for retrieve+execute) and
# safety_output runs AFTER reflect (last line of defense: a clean prompt
# with a bad output still gets blocked).
#
# SAFETY_FILTER_ENABLED is the master switch. When false, both safety
# nodes pass through (return safety_action="disabled") without making
# any HTTP call. Default false so the graph topology can land
# independently of operator-side decisions about scaling up Llama Guard.
# Flip to true via the langgraph-service Deployment env once the
# vllm-llama-guard-3-8b pod is reachable.
SAFETY_FILTER_ENABLED = os.environ.get("SAFETY_FILTER_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
SAFETY_LLAMA_GUARD_URL = os.environ.get(
    "SAFETY_LLAMA_GUARD_URL",
    "http://vllm-llama-guard-3-8b.llm.svc.cluster.local:8000/v1",
)
SAFETY_LLAMA_GUARD_MODEL = os.environ.get(
    "SAFETY_LLAMA_GUARD_MODEL", "llama-guard-3-8b"
)
SAFETY_TIMEOUT_SECONDS = float(os.environ.get("SAFETY_TIMEOUT_SECONDS", "15"))

# Fail-open vs fail-closed when Llama Guard is unreachable or returns
# malformed output. fail-open = allow the request (safety filter degrades
# to no-op on infra issues); fail-closed = block (safety > availability).
# Default open for the lab — production deployments invert this.
SAFETY_FAIL_MODE = os.environ.get("SAFETY_FAIL_MODE", "open").lower()  # "open" | "closed"

# Hazard categories that trigger a block. Default = all 14 from Llama
# Guard 3's taxonomy. Operators can narrow via env (e.g.
# "S1,S4,S9,S10,S11" to block only the most severe). Empty string =
# block any unsafe verdict regardless of category.
SAFETY_BLOCK_CATEGORIES = set(
    c.strip().upper()
    for c in os.environ.get(
        "SAFETY_BLOCK_CATEGORIES",
        "S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14",
    ).split(",")
    if c.strip()
)

# What we tell the user when their input/output gets blocked. Vague
# rather than specific to avoid leaking which category triggered (a
# probing attacker would otherwise iterate to find phrasings that pass).
SAFETY_REFUSAL_MESSAGE = os.environ.get(
    "SAFETY_REFUSAL_MESSAGE",
    "I can't help with that. Please rephrase your request or ask about something else.",
)

# --- Phase #5: cost guardrails (Redis token bucket) ------------------------
#
# Per-user daily request budget. Each /invoke consumes 1 credit; the
# user is rate-limited to BUDGET_REQUESTS_PER_DAY total credits per UTC
# day. Backed by a Redis key per (user, date) with a 48h TTL — daily
# rollover is automatic (a new day uses a new key, the previous day's
# key expires on its own), no CronJob needed.
#
# Where this sits in the graph: node_budget_check is the FIRST node
# after START, even before safety_input. Two reasons:
#   1. Cheapest possible check (one Redis INCR), so it should be first.
#   2. Abuse-prevention: an attacker probing the safety filter for
#      false-positive boundaries would otherwise get unlimited free
#      attempts. Charging upfront makes probing costly.
# This does mean a legitimate user who hits a safety false-positive
# still consumes a credit. Acceptable trade for the lab; an operator
# who wants fairer-to-users posture can swap the order in build_graph.
#
# Default disabled — the BUDGET_REQUESTS_PER_DAY env defaults to 0
# which the runner interprets as "filter is off, pass through every
# request". Flip on by setting BUDGET_ENABLED=true in deployment.yaml
# AND BUDGET_REQUESTS_PER_DAY to a positive integer.
BUDGET_ENABLED = os.environ.get("BUDGET_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
BUDGET_REDIS_URL = os.environ.get(
    "BUDGET_REDIS_URL", "redis://langgraph-redis.langgraph.svc.cluster.local:6379/0"
)
BUDGET_REQUESTS_PER_DAY = int(os.environ.get("BUDGET_REQUESTS_PER_DAY", "200"))
BUDGET_TIMEOUT_SECONDS = float(os.environ.get("BUDGET_TIMEOUT_SECONDS", "2"))
# Same fail-mode pattern as the safety filter. open = if Redis is
# unreachable, allow the request (rate limiting degrades to no-op).
# closed = block. Lab default is open; production sets closed.
BUDGET_FAIL_MODE = os.environ.get("BUDGET_FAIL_MODE", "open").lower()
BUDGET_REFUSAL_MESSAGE = os.environ.get(
    "BUDGET_REFUSAL_MESSAGE",
    "Daily request budget exhausted. Try again after UTC midnight.",
)

# --- Phase #6: conversational memory + query rewriting --------------------
#
# Two coupled features that compound on each other:
#
#   load_memory:   reads recent conversation turns + long-term summary
#                  from Redis (keyed per user + session_id).
#   rewrite_query: takes the raw user prompt + conversation context and
#                  produces a standalone search query — replaces pronouns,
#                  resolves "it"/"that" against the conversation, preserves
#                  technical terms verbatim. Sets state.refined_query so
#                  node_retrieve uses it (the Phase T2 mechanism).
#   save_memory:   appends the (prompt, response) turn at the end of the
#                  graph, re-summarizes every N turns to bound the
#                  recent-turns list size.
#
# Why per-(user, session) keying and not just per-session: prevents any
# accidental cross-user leakage if two users somehow shared a session_id
# (chat-ui generates UUIDs so it's unlikely, but the cost is one extra
# string in the Redis key).
#
# 7-day TTL on memory keys gives a sensible idle-session expiry. Manual
# right-to-deletion is `redis-cli --scan --pattern 'mem:<user>:*' | xargs
# redis-cli del`.
#
# Reuses the same Redis client (_get_redis from Phase #5) — different
# key prefix (mem: vs cost:), same connection pool.
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
MEMORY_RECENT_TURNS = int(os.environ.get("MEMORY_RECENT_TURNS", "5"))
MEMORY_SUMMARIZE_AFTER_TURNS = int(
    os.environ.get("MEMORY_SUMMARIZE_AFTER_TURNS", "10")
)
MEMORY_TTL_SECONDS = int(os.environ.get("MEMORY_TTL_SECONDS", str(7 * 86400)))
MEMORY_TIMEOUT_SECONDS = float(os.environ.get("MEMORY_TIMEOUT_SECONDS", "2"))

QUERY_REWRITE_ENABLED = os.environ.get("QUERY_REWRITE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
QUERY_REWRITE_MAX_TOKENS = int(os.environ.get("QUERY_REWRITE_MAX_TOKENS", "96"))
QUERY_REWRITE_TIMEOUT_SECONDS = float(
    os.environ.get("QUERY_REWRITE_TIMEOUT_SECONDS", "10")
)


def _memory_turns_key(user: str, session_id: str) -> str:
    return f"mem:{user}:{session_id}:turns"


def _memory_summary_key(user: str, session_id: str) -> str:
    return f"mem:{user}:{session_id}:summary"

# Lazy-initialized at first use. Reused across requests via the same
# connection-pooled client. redis-py is thread-safe for this usage.
_REDIS_CLIENT: Optional["redis.Redis"] = None  # type: ignore[name-defined]


def _get_redis():
    """Return the lazy-initialized redis client, or None on connection failure.

    None signals "treat as unavailable" — node_budget_check then
    consults BUDGET_FAIL_MODE. We don't raise from import time because
    Redis being briefly unavailable shouldn't crash the pod.
    """
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    try:
        import redis as _redis_pkg
        client = _redis_pkg.Redis.from_url(
            BUDGET_REDIS_URL,
            socket_timeout=BUDGET_TIMEOUT_SECONDS,
            socket_connect_timeout=BUDGET_TIMEOUT_SECONDS,
            decode_responses=True,
        )
        # Don't ping at construction — let the first INCR attempt
        # discover unreachability and fall through to fail-mode. Saves
        # a round-trip on every cold pod start.
        _REDIS_CLIENT = client
        return _REDIS_CLIENT
    except Exception as e:
        log.warning("redis client init failed: %s", e)
        return None


def _budget_today_key(username: str) -> str:
    """Deterministic daily key. UTC date avoids local-zone surprises.

    Form: cost:<username>:<YYYY-MM-DD>. The colon-delimited shape is
    redis-cli-greppable (KEYS cost:raj:* shows raj's whole history)
    and survives the 48h TTL window so an audit query at 4 AM UTC the
    next day still sees yesterday's counter alongside today's.
    """
    from datetime import datetime, timezone
    return f"cost:{username}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

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


# --- Reflection (Phase T2) -------------------------------------------------
#
# After node_execute drafts an answer, node_reflect asks the cheap
# always-on Llama 8B whether running ANOTHER retrieval (with a refined
# query) would meaningfully improve the answer. If yes, the conditional
# edge after this node loops back to retrieve → ensure_warm → execute →
# reflect, capped at MAX_REASONING_CYCLES total cycles.
#
# Why a *separate* reflection model call instead of reading the agent's
# tool-call output? Two reasons:
#   1. The agent's tool calls inside node_execute happen during a single
#      LLM exchange — once the agent commits to a final-text answer in
#      that exchange, it's done. Graph-level reflection lets us SECOND-
#      GUESS the final answer with a fresh perspective.
#   2. Some routes (reasoning, hard) don't support tools at all. The
#      reflect node only fires for tool-capable tiers (gated below) so
#      DeepSeek-R1 / 70B don't get a free reflection budget they didn't
#      sign up for.

REFLECT_SYSTEM_PROMPT = """You decide whether a draft answer needs more research.

Read the question, the draft answer, and the list of sources already searched. Decide if running ONE MORE document search (with a different query) would meaningfully improve the answer.

Output exactly one line of valid JSON, no prose:
  {"needs_more": true, "query": "<a NEW search query different from prior ones>"}
  {"needs_more": false}

Choose `needs_more: true` ONLY when:
  - the draft answer admits it doesn't know something the user asked
  - the draft answer references a concept the sources didn't cover
  - the question has multiple parts and the draft only addressed some

Choose `needs_more: false` when:
  - the draft answer is already a complete response
  - more searching wouldn't help (e.g. it's a math/logic answer, or the missing info isn't in any docs)

Output ONLY the JSON. No code fences, no explanation."""


def node_reflect(state: AgentState) -> AgentState:
    """T2: graph-level reasoning loop decision node.

    Increments the cycle counter, then calls Llama 8B to decide whether
    a follow-up retrieval cycle would help. Output drives the conditional
    edge (`_route_after_reflect`) — either back to `retrieve` for another
    pass, or terminate the graph.

    Skips reflection (immediately decides "no") when:
      - The route doesn't support tools — non-tool tiers don't
        participate in graph-level reasoning loops.
      - Cycle cap reached.
      - No session_id — without a session there's nothing to re-retrieve.
      - The reflection model returns malformed JSON or the call fails.
        Fail-safe: assume the answer is complete (don't loop).
    """
    cycles = state.get("cycles", 0) + 1
    reflection_log = list(state.get("reflection_log", []) or [])

    cfg = ROUTE_REGISTRY[state["route"]]
    if not cfg.get("supports_tools"):
        reflection_log.append(f"cycle {cycles}: route={state['route']} doesn't support tools, skipping reflection")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    if cycles >= MAX_REASONING_CYCLES:
        reflection_log.append(f"cycle {cycles}: cap reached (MAX_REASONING_CYCLES={MAX_REASONING_CYCLES})")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    if not state.get("session_id"):
        reflection_log.append(f"cycle {cycles}: no session_id, nothing to re-retrieve")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    cfg_trivial = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg_trivial["model_name"],
        base_url=cfg_trivial["url"],
        api_key="not-required",
        temperature=REFLECT_TEMPERATURE,
        max_tokens=REFLECT_MAX_TOKENS,
        timeout=REFLECT_TIMEOUT_SECONDS,
    )

    chunks = state.get("retrieved_chunks") or []
    sources_summary = (
        ", ".join(
            f"[{i + 1}] {c.get('source', 'unknown')} (chunk {c.get('chunk_index', 0)})"
            for i, c in enumerate(chunks)
        )
        or "none"
    )
    # Cap the draft snippet — Llama 8B's context budget for the
    # reflection step is tight (32 max_tokens output, ~2k input). The
    # first 1000 chars of the draft is enough to judge completeness.
    draft = (state.get("response", "") or "")[:1000]

    user_msg = (
        f"Original question:\n{state['prompt']}\n\n"
        f"Sources already searched:\n{sources_summary}\n\n"
        f"Draft answer:\n{draft}"
    )

    try:
        resp = client.invoke(
            [
                SystemMessage(content=REFLECT_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        raw = (resp.content or "").strip()
    except Exception as e:
        # Fail-safe: don't loop on reflection errors. Better to return
        # the current draft than 502 the whole request.
        log.warning("reflect: model call failed (treating as complete): %s", e)
        reflection_log.append(f"cycle {cycles}: reflect call failed: {e}")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    # Parse the first JSON object from the response. Llama 8B sometimes
    # wraps the JSON in code fences or trailing newlines; the regex
    # finds the first {...} block regardless.
    import json as _json
    import re as _re

    match = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
    if not match:
        reflection_log.append(f"cycle {cycles}: no JSON in reflect output={raw[:120]!r}")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    try:
        decision = _json.loads(match.group(0))
    except _json.JSONDecodeError:
        reflection_log.append(f"cycle {cycles}: malformed JSON={match.group(0)[:120]!r}")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    needs = bool(decision.get("needs_more"))
    refined = (decision.get("query") or "").strip()

    if not needs:
        reflection_log.append(f"cycle {cycles}: answer complete")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    if not refined:
        # Model said yes but didn't give a query. Treat as no — looping
        # to retrieve with the original query would just find the same
        # chunks (which we already have).
        reflection_log.append(f"cycle {cycles}: needs_more=true but no query supplied; stopping")
        return {
            "cycles": cycles,
            "needs_more_context": False,
            "reflection_log": reflection_log,
        }

    reflection_log.append(f"cycle {cycles}: looping back to retrieve, refined_query={refined!r}")
    log.info("reflect: cycle=%d looping with refined_query=%r", cycles, refined[:80])
    return {
        "cycles": cycles,
        "needs_more_context": True,
        "refined_query": refined,
        "reflection_log": reflection_log,
    }


def _route_after_reflect(state: AgentState) -> Literal["retrieve", "__end__"]:
    """Conditional edge function for the reflect node.

    Returns the name of the next node. LangGraph maps this to the
    `path_map` arg of add_conditional_edges. The cycle cap check is
    duplicated here (also enforced inside node_reflect) so a buggy
    reflect node can never produce an infinite loop — the conditional
    edge enforces the bound regardless.
    """
    if state.get("needs_more_context") and state.get("cycles", 0) < MAX_REASONING_CYCLES:
        return "retrieve"
    return END


def build_graph() -> StateGraph:
    """Compile the seven-node graph with safety bookends + reasoning loop.

    Linear path:
      START → safety_input → classify → retrieve → ensure_warm
            → execute → reflect → safety_output → END

    Conditional edges:
      safety_input → END (if blocked) or classify
      reflect      → retrieve (loop, capped) or safety_output

    The retrieve node is cycle-aware: on first entry, query = prompt;
    on re-entry from reflect, query = refined_query. Chunks accumulate
    (deduped) across cycles. Cap is MAX_REASONING_CYCLES.

    Safety nodes pass through (return safety_action="passed") when
    SAFETY_FILTER_ENABLED is false, so the same graph topology applies
    regardless of operator-side enablement decisions. This keeps Langfuse
    + OTel traces consistent across enabled/disabled runs.
    """
    g: StateGraph = StateGraph(AgentState)
    g.add_node("budget_check", node_budget_check)
    g.add_node("safety_input", node_safety_input)
    g.add_node("load_memory", node_load_memory)
    g.add_node("rewrite_query", node_rewrite_query)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("ensure_warm", node_ensure_warm)
    g.add_node("execute", node_execute)
    g.add_node("reflect", node_reflect)
    g.add_node("safety_output", node_safety_output)
    g.add_node("save_memory", node_save_memory)
    g.add_edge(START, "budget_check")
    g.add_conditional_edges(
        "budget_check",
        _route_after_budget_check,
        {"safety_input": "safety_input", END: END},
    )
    g.add_conditional_edges(
        "safety_input",
        _route_after_safety_input,
        # Now routes to load_memory rather than directly to classify so
        # the rewrite_query node has conversation context to work with.
        {"classify": "load_memory", END: END},
    )
    g.add_edge("load_memory", "rewrite_query")
    g.add_edge("rewrite_query", "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "ensure_warm")
    g.add_edge("ensure_warm", "execute")
    g.add_edge("execute", "reflect")
    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"retrieve": "retrieve", END: "safety_output"},
    )
    # safety_output → save_memory → END so the persisted turn is the
    # text the user actually saw (refusal text on block, real answer
    # otherwise). save_memory itself short-circuits on safety-blocked
    # or budget-blocked paths so failed attempts don't pollute history.
    g.add_edge("safety_output", "save_memory")
    g.add_edge("save_memory", END)
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
    # Phase T2: graph-level reasoning loop telemetry.
    # reasoning_cycles is the count of full retrieve→execute→reflect
    # passes the graph completed (1 for non-loop runs, up to
    # MAX_REASONING_CYCLES for tool-capable tiers that re-retrieved).
    # reflection_log is one human-readable line per cycle describing
    # the reflect decision — useful for debugging "why did the agent
    # loop?" questions and for Langfuse correlation.
    reasoning_cycles: int = 0
    reflection_log: list[str] = []
    # Phase #4: content safety telemetry. action is the terminal
    # disposition; verdicts are per-node Llama Guard outputs (or
    # "skipped"/"disabled"/"fail_*" for non-flowing paths). categories
    # is the union of S-codes that triggered any block (empty for safe
    # or disabled flows). Surfacing latencies separately so chat-ui
    # and Langfuse can split a slow request between the safety filter
    # vs the model itself.
    safety_action: str = "passed"
    safety_input_verdict: str = "skipped"
    safety_output_verdict: str = "skipped"
    safety_categories: list[str] = []
    safety_input_ms: int = 0
    safety_output_ms: int = 0
    # Phase #5: cost-guardrail telemetry. action drives chat-ui badge
    # rendering (budget exhausted vs filter disabled vs passed). The
    # remaining counter lets the UI render "X requests left today" so
    # users see budget pressure before they hit the wall.
    budget_action: str = "disabled"
    budget_consumed: int = 0
    budget_remaining: int = 0
    # Phase #6: memory + query rewriting telemetry. query_rewritten
    # is the standalone search query (set when rewrite ran AND
    # produced something different from the original prompt; empty
    # otherwise). memory_turn_count helps chat-ui render "you've
    # had N turns in this session" widgets.
    query_rewritten: str = ""
    query_rewrite_ms: int = 0
    memory_turn_count: int = 0
    memory_load_ms: int = 0
    memory_save_ms: int = 0


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
        reasoning_cycles=final_state.get("cycles", 0),
        reflection_log=final_state.get("reflection_log", []),
        safety_action=final_state.get("safety_action", "passed"),
        safety_input_verdict=final_state.get("safety_input_verdict", "skipped"),
        safety_output_verdict=final_state.get("safety_output_verdict", "skipped"),
        safety_categories=final_state.get("safety_categories", []),
        safety_input_ms=final_state.get("safety_input_ms", 0),
        safety_output_ms=final_state.get("safety_output_ms", 0),
        budget_action=final_state.get("budget_action", "disabled"),
        budget_consumed=final_state.get("budget_consumed", 0),
        budget_remaining=final_state.get("budget_remaining", 0),
        query_rewritten=(
            final_state.get("refined_query", "") or ""
            if final_state.get("query_rewritten") else ""
        ),
        query_rewrite_ms=final_state.get("query_rewrite_ms", 0),
        memory_turn_count=len(final_state.get("memory_recent_turns") or []),
        memory_load_ms=final_state.get("memory_load_ms", 0),
        memory_save_ms=final_state.get("memory_save_ms", 0),
    )
