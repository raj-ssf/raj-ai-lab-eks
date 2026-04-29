"""langgraph-service — FastAPI front-end for a LangGraph router agent.

Eighteen-node state machine on /invoke (budget + input validation +
safety + input/output PII redaction + cache + memory + query rewriting
+ planning + reasoning loop + hallucination check):

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
- input_validation: regex/string defense layer for attack classes
  Llama Guard isn't trained on (homoglyph, zero-width smuggle,
  context-window exhaustion, control chars, whitespace padding).
  Two outcomes: normalize (in-place sanitize) or block (short-
  circuit). Runs before safety_input so Llama Guard's 300ms call
  isn't wasted on malformed input. No-op when
  VALIDATE_INPUT_ENABLED=false.
- safety_input: Llama Guard 3 8B grades the user's prompt against
  Meta's 14-category hazard taxonomy. If unsafe AND the violated
  categories intersect SAFETY_BLOCK_CATEGORIES, the graph short-
  circuits to END with a refusal response. No-op when
  SAFETY_FILTER_ENABLED=false.
- pii_redact_input: regex-based PII detection over the user's prompt
  (Phase #16). Produces state.redacted_prompt used by cache_lookup,
  retrieve, rewrite_query, save_memory. Original prompt still goes
  to node_execute (model needs the actual content). No-op when
  PII_REDACT_INPUT_ENABLED=false.
- cache_lookup: embeds the prompt via vllm-bge-m3 and finds the most-
  similar cached entry for (user, session_id). On hit (cosine ≥
  CACHE_SIMILARITY_THRESHOLD), the cached response is returned and
  the entire downstream pipeline is skipped. No-op when CACHE_ENABLED=
  false. Fail-OPEN on bge-m3 or Redis errors → cache miss.
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
- plan: per-tier opt-in plan-and-execute pattern. Llama 8B produces a
  numbered list of steps (TOOL/REASON/RESPOND) before execute runs.
  The plan becomes additional system context for the agentic execute
  loop — guidance, not a script. Per-tier via
  ROUTE_REGISTRY[tier].use_planner; off for trivial, on for
  reasoning/hard. Globally gated by PLANNER_ENABLED.
- execute: HTTP POST to the chosen variant's vLLM OpenAI-compat endpoint.
  For tool-capable tiers, runs an agentic tool-call loop bounded by
  AGENT_MAX_ITERATIONS (per-execute, distinct from the graph-level cap).
- reflect: gates whether to loop. Asks the cheap Llama 8B to decide if
  another retrieval cycle (with a new query) would meaningfully improve
  the draft answer. If yes AND we haven't hit MAX_REASONING_CYCLES,
  routes back to retrieve; otherwise routes to safety_output.
- safety_output: Llama Guard 3 8B grades the model's draft. If unsafe,
  replaces `response` with the refusal message before END.
- hallucination_check: Llama 8B grades whether the response is
  grounded in the retrieved chunks. Verdicts: grounded, partial,
  ungrounded. Action depends on HALLUCINATION_ACTION (flag = record
  only; block = prepend a confidence disclaimer). Skips on cache hit,
  no chunks, safety blocked. No-op when HALLUCINATION_CHECK_ENABLED=
  false.
- pii_redact_output: regex-based PII detection (email, phone, SSN,
  credit card, IPv4, AWS access key) over the response, replacing
  matched spans with <redacted_TYPE>. Telemetry surfaces counts by
  entity type — never the original values. No-op when
  PII_REDACT_OUTPUT_ENABLED=false.
- cache_store: persists the safety-checked response to the prompt
  cache for future similar requests. LRU-evicts to keep cache size
  bounded. Skips on safety-blocked, budget-blocked, or cache-hit
  paths.
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
from fastapi.responses import StreamingResponse
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
from prometheus_client import Counter, Histogram
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
        # Phase #13: plan-and-execute opt-in. Trivial tier prompts are
        # typically single-step ("what's 2+2") so planning would add
        # ~500ms of LLM call latency for marginal benefit. Default off.
        # Operators wanting to demo plan-and-execute on this tier flip
        # to True via a deployment.yaml env override (per-tier flags
        # are in code; the master switch PLANNER_ENABLED gates all).
        "use_planner": False,
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
        "use_planner": False,
    },
    "reasoning": {
        "url": f"{MODEL_REASONING_URL}/v1",
        "model_name": MODEL_REASONING_NAME,
        "deployment": DEPLOY_REASONING,
        "always_on": False,
        "supports_tools": False,  # DeepSeek-R1 70B's chat-template support TBD
        # Reasoning tier handles complex multi-step queries — planning
        # helps most here. On by default at the tier level (still
        # gated by master PLANNER_ENABLED).
        "use_planner": True,
    },
    "hard": {
        "url": f"{MODEL_HARD_URL}/v1",
        "model_name": MODEL_HARD_NAME,
        "deployment": DEPLOY_HARD,
        "always_on": False,
        "supports_tools": False,  # 70B AWQ tier; tool support TBD
        # Hard tier serves long-form / complex tasks. Planning helps.
        "use_planner": True,
    },
}


# --- Phase #15: canary variant routing per tier ----------------------------
#
# Per-tier env-driven canary: route a configurable fraction of requests
# to an alternate model name (typically a fine-tuned variant served on
# the same vLLM pod). The lab's vllm-llama-8b serves both "llama-3.1-8b"
# (base) and "llama-3.1-8b-alpaca" (LoRA) — natural pair for canary.
#
# Per request, after classify picks a tier, _select_variant flips a
# weighted coin to decide stable vs canary. The chosen model_name is
# recorded in state.variant_name + state.variant_label and used by
# node_execute. Both /invoke responses and the new LG_VARIANT_TOTAL
# Counter expose the variant label for downstream A/B analysis.
#
# Env config shape:
#   CANARY_<TIER>_MODEL=<model_name>     # alternate served-model-name
#   CANARY_<TIER>_FRACTION=<0.0-1.0>     # request fraction to canary
#
# Default fraction is 0.0 — no canary unless explicitly configured.
# Set per tier: CANARY_TRIVIAL_MODEL=llama-3.1-8b-alpaca,
# CANARY_TRIVIAL_FRACTION=0.1 to canary 10% of trivial traffic to
# the LoRA-merged variant.

CANARY_CONFIG: dict[str, dict] = {}
for _tier in ROUTE_REGISTRY.keys():
    _model = os.environ.get(f"CANARY_{_tier.upper().replace('-','_')}_MODEL", "").strip()
    _frac_raw = os.environ.get(f"CANARY_{_tier.upper().replace('-','_')}_FRACTION", "0").strip()
    try:
        _frac = max(0.0, min(1.0, float(_frac_raw)))
    except ValueError:
        _frac = 0.0
    if _model and _frac > 0.0:
        CANARY_CONFIG[_tier] = {"model": _model, "fraction": _frac}


def _select_variant(route: str) -> tuple[str, str]:
    """Return (model_name, variant_label) for a routed-to tier.

    variant_label is "stable" when the request goes to the tier's
    default model_name, "canary" when the canary roll triggered. Used
    for telemetry + downstream A/B analysis.
    """
    cfg = ROUTE_REGISTRY[route]
    canary = CANARY_CONFIG.get(route)
    if not canary:
        return (cfg["model_name"], "stable")
    import random as _random
    if _random.random() < canary["fraction"]:
        return (canary["model"], "canary")
    return (cfg["model_name"], "stable")

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


# --- Phase #14: custom Prometheus metrics ----------------------------------
#
# The Instrumentator above auto-instruments HTTP-level metrics (request
# count + duration histogram by path/status). These additional Counter
# and Histogram metrics expose the per-node telemetry the graph
# accumulates — safety blocks, cache hits, hallucination verdicts, etc.
# — so Grafana dashboards can plot them.
#
# Naming: langgraph_<concern>_<unit> is the convention. Counters end in
# _total per Prometheus best practice; Histograms in _<unit>.
#
# Emission point: a single _emit_request_metrics(final_state) helper
# reads the terminal state at the end of /invoke and /invoke/stream
# and increments/observes the relevant series. Keeping the metric
# emission centralized — rather than scattered across every node —
# means the metrics surface is reviewable in one place and stays in
# sync with the InvokeResponse shape.

LG_REQUEST_TOTAL = Counter(
    "langgraph_requests_total",
    "Total /invoke requests by routed-to tier.",
    ["route"],
)
# Phase #15: per-tier per-variant breakdown for A/B canary analysis.
# Separate Counter (rather than adding a label to LG_REQUEST_TOTAL)
# preserves backwards-compat for existing dashboards while letting
# new A/B-specific panels query this series.
LG_VARIANT_TOTAL = Counter(
    "langgraph_variant_total",
    "Per-tier per-variant request count for A/B canary analysis.",
    ["route", "variant"],
)
LG_SAFETY_ACTION_TOTAL = Counter(
    "langgraph_safety_action_total",
    "Terminal safety_action disposition per request.",
    ["action"],
)
LG_CACHE_ACTION_TOTAL = Counter(
    "langgraph_cache_action_total",
    "Cache outcome — hit, miss, or disabled.",
    ["action"],
)
LG_BUDGET_ACTION_TOTAL = Counter(
    "langgraph_budget_action_total",
    "Per-user budget gate disposition.",
    ["action"],
)
LG_HALLUCINATION_ACTION_TOTAL = Counter(
    "langgraph_hallucination_action_total",
    "Hallucination check outcome.",
    ["verdict"],
)
LG_PLANNER_ACTION_TOTAL = Counter(
    "langgraph_planner_action_total",
    "Planner outcome — planned, skipped, fail_open.",
    ["action"],
)
LG_PII_REDACT_ACTION_TOTAL = Counter(
    "langgraph_pii_redact_action_total",
    "PII redaction outcome.",
    ["action"],
)
LG_TOOL_CALLS_TOTAL = Counter(
    "langgraph_tool_calls_total",
    "Tool invocations from the agentic execute loop, by tool name.",
    ["tool"],
)
# Phase #19: per-tool rate-limit denials. Spikes here indicate either
# legitimate overuse (raise the limit) or actual abuse (investigate
# the user). Operators can alert on this rate.
LG_TOOL_RATE_LIMITED_TOTAL = Counter(
    "langgraph_tool_rate_limited_total",
    "Per-tool rate-limit denials, by tool name.",
    ["tool"],
)
# Phase #20: input validation outcomes. Spikes on a specific reason
# (length_exceeded, control_chars, excessive_whitespace) suggest
# automated probing. Operators should alert on rate(reason=length).
LG_INPUT_VALIDATION_TOTAL = Counter(
    "langgraph_input_validation_total",
    "Outcomes of node_input_validation, by action and reason.",
    ["action", "reason"],
)
LG_REASONING_CYCLES = Histogram(
    "langgraph_reasoning_cycles",
    "Graph-level reasoning loop cycles per request (Phase T2).",
    buckets=(0, 1, 2, 3, 4, 5),
)
# Per-node duration histogram. One series per node label keeps the
# cardinality bounded (16 nodes × 8 buckets = 128 distinct series),
# which Prometheus handles fine.
LG_NODE_DURATION_SECONDS = Histogram(
    "langgraph_node_duration_seconds",
    "Per-node graph execution duration in seconds.",
    ["node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# State-field → metric-label-friendly node name mapping.
_NODE_MS_FIELDS: dict[str, str] = {
    "safety_input": "safety_input_ms",
    "safety_output": "safety_output_ms",
    "cache_lookup": "cache_lookup_ms",
    "cache_store": "cache_store_ms",
    "memory_load": "memory_load_ms",
    "memory_save": "memory_save_ms",
    "query_rewrite": "query_rewrite_ms",
    "hallucination_check": "hallucination_check_ms",
    "pii_redact_input": "pii_input_redact_ms",
    "pii_redact_output": "pii_redact_ms",
    "plan": "plan_ms",
    "execute": "execute_latency_ms",
    "retrieve": "retrieve_ms",
}


def _emit_request_metrics(final_state: dict) -> None:
    """Increment Counters + observe Histograms from a terminal graph state.

    Called from /invoke (sync) at the end of the handler and from
    /invoke/stream's done-event branch. Best-effort — exceptions in
    metric emission are caught + logged so a buggy metric doesn't
    fail the request.
    """
    try:
        LG_REQUEST_TOTAL.labels(route=final_state.get("route", "unknown")).inc()
        LG_SAFETY_ACTION_TOTAL.labels(
            action=final_state.get("safety_action", "passed")
        ).inc()

        # Cache: derive a 3-way action (hit/miss/disabled) — the state
        # only carries cache_hit (bool) + cache_lookup_ms (0 if disabled).
        if final_state.get("cache_hit"):
            cache_label = "hit"
        elif final_state.get("cache_lookup_ms", 0) == 0:
            cache_label = "disabled"
        else:
            cache_label = "miss"
        LG_CACHE_ACTION_TOTAL.labels(action=cache_label).inc()

        LG_BUDGET_ACTION_TOTAL.labels(
            action=final_state.get("budget_action", "disabled")
        ).inc()
        LG_HALLUCINATION_ACTION_TOTAL.labels(
            verdict=final_state.get("hallucination_verdict", "skipped")
        ).inc()
        LG_PLANNER_ACTION_TOTAL.labels(
            action=final_state.get("planner_action", "skipped")
        ).inc()
        LG_PII_REDACT_ACTION_TOTAL.labels(
            action=final_state.get("pii_redact_action", "skipped")
        ).inc()

        for tool in final_state.get("tool_calls_log") or []:
            LG_TOOL_CALLS_TOTAL.labels(tool=tool).inc()

        cycles = final_state.get("cycles", 0) or 0
        LG_REASONING_CYCLES.observe(cycles)

        # Per-node durations
        for node_label, ms_field in _NODE_MS_FIELDS.items():
            ms = final_state.get(ms_field, 0) or 0
            if ms > 0:
                LG_NODE_DURATION_SECONDS.labels(
                    node=node_label
                ).observe(ms / 1000.0)
    except Exception as e:
        log.warning("metric emission failed: %s", e)

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
    # Phase #7: semantic prompt cache bookkeeping.
    #   cache_hit: True if cache_lookup found a match above threshold.
    #     When True, downstream graph (load_memory through safety_output)
    #     is skipped via the conditional edge after cache_lookup.
    #   cache_similarity: cosine similarity of the best match (range
    #     0.0-1.0). Surfaced for observability — operators can tune
    #     CACHE_SIMILARITY_THRESHOLD by watching this distribution.
    #   prompt_embedding: 1024-dim bge-m3 embedding of the prompt.
    #     Computed once in cache_lookup, reused in cache_store to
    #     avoid a second embedding call on cache misses.
    #   cache_lookup_ms / cache_store_ms: per-node latencies.
    cache_hit: bool
    cache_similarity: float
    prompt_embedding: Optional[list[float]]
    cache_lookup_ms: int
    cache_store_ms: int
    # Phase #9: runtime hallucination detection bookkeeping.
    #   hallucination_verdict: "grounded" | "partial" | "ungrounded" |
    #     "skipped" | "fail_open"
    #   hallucination_confidence: 0.0–1.0 from the grader's self-reported
    #     confidence in its verdict
    #   hallucination_action: terminal disposition — "passed" |
    #     "flagged" | "blocked" | "disabled"
    #   hallucination_check_ms: per-node latency
    hallucination_verdict: Literal["grounded", "partial", "ungrounded", "skipped", "fail_open"]
    hallucination_confidence: float
    hallucination_action: Literal["passed", "flagged", "blocked", "disabled"]
    hallucination_check_ms: int
    # Phase #11: PII redaction telemetry.
    #   pii_redact_action: terminal disposition — "redacted" |
    #     "passed" (no PII found) | "skipped" (filter disabled / cache
    #     hit / safety blocked).
    #   pii_entities_found: dict mapping entity_type -> count. NEVER
    #     contains the original PII values themselves — that would
    #     defeat the redaction by leaking via API.
    #   pii_redact_ms: per-node latency.
    pii_redact_action: Literal["redacted", "passed", "skipped"]
    pii_entities_found: dict
    pii_redact_ms: int
    # Phase #16: input-side PII redaction.
    #   redacted_prompt: the user's prompt with PII replaced by
    #     <redacted_TYPE> placeholders. Used by cache_lookup, retrieve,
    #     rewrite_query, save_memory. Falls back to state.prompt when
    #     filter is disabled or no PII found.
    #   pii_input_action / _entities_found / _ms: mirror output-side
    #     telemetry shape.
    redacted_prompt: str
    pii_input_action: Literal["redacted", "passed", "skipped"]
    pii_input_entities_found: dict
    pii_input_redact_ms: int
    # Phase #13: planner output.
    #   plan_text: raw planner LLM output (numbered list of steps,
    #     human-readable). Empty when planner skipped.
    #   plan_steps_count: parsed count of steps the planner emitted.
    #     Bounded by PLANNER_MAX_STEPS in the prompt; runtime parses
    #     for telemetry, not for control flow.
    #   planner_action: terminal disposition — "planned" | "skipped"
    #     | "fail_open" (planner LLM call failed).
    #   plan_ms: per-node latency.
    plan_text: str
    plan_steps_count: int
    planner_action: Literal["planned", "skipped", "fail_open"]
    plan_ms: int
    # Phase #15: A/B canary routing.
    #   variant_name: the model name node_execute actually used. Equal
    #     to ROUTE_REGISTRY[route].model_name on stable requests; equal
    #     to CANARY_<TIER>_MODEL when the canary roll triggered.
    #   variant_label: "stable" or "canary". Surfaced in metrics +
    #     InvokeResponse so downstream eval tooling can correlate
    #     ratings, latencies, etc. by variant.
    variant_name: str
    variant_label: Literal["stable", "canary"]
    # Phase #19: per-tool rate-limit log. Names of tools the agent
    # tried to call but were rate-limited (per-(tool, user) sliding
    # window). Surfaced for chat-ui to render "your http_fetch usage
    # exceeded the limit" messages.
    tool_rate_limited_log: list[str]
    # Phase #20: input validation outcome.
    #   action: "passed" | "normalized" | "blocked" | "skipped"
    #   details: per-check fields (zero_width_stripped count,
    #     unicode_normalized bool, length on length_exceeded, etc.)
    #     NEVER contains the raw matched content — would defeat the
    #     point of the filter.
    input_validation_action: Literal["passed", "normalized", "blocked", "skipped"]
    input_validation_details: dict


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


def _route_after_budget_check(state: AgentState) -> Literal["input_validation", "__end__"]:
    """Conditional edge: if budget exceeded or fail_closed, terminate.

    Defense-in-depth pattern matches _route_after_safety_input — even
    if a future edit produces an "exceeded" action without setting
    response, this edge still routes correctly to END.

    Phase #20: now routes to input_validation (interposed before
    safety_input) on the happy path. Validation is cheap (no I/O,
    no LLM call) so it appropriately runs before Llama Guard's
    300ms safety call on adversarial input.
    """
    if state.get("budget_action") in ("exceeded", "fail_closed"):
        return END
    return "input_validation"


def node_input_validation(state: AgentState) -> AgentState:
    """Phase #20: validate + sanitize the user's prompt.

    Three outcomes:
      passed       no issues; proceed unchanged
      normalized   defensive fixes applied (NFKC + zero-width strip);
                   state.prompt overwritten with sanitized version,
                   downstream nodes see the cleaned text
      blocked      hard violation (length, control chars, excessive
                   whitespace) — pre-populate response, conditional
                   edge routes to END

    Skip when VALIDATE_INPUT_ENABLED=false. No-op fallthrough.
    """
    if not VALIDATE_INPUT_ENABLED:
        LG_INPUT_VALIDATION_TOTAL.labels(action="skipped", reason="").inc()
        return {
            "input_validation_action": "skipped",
            "input_validation_details": {},
        }

    prompt = state.get("prompt", "") or ""
    sanitized, action, details = _validate_input(prompt)

    if action == "blocked":
        reason = details.get("reason", "unknown")
        LG_INPUT_VALIDATION_TOTAL.labels(action="blocked", reason=reason).inc()
        log.info(
            "input_validation blocked reason=%s details=%s",
            reason, details,
        )
        return {
            "input_validation_action": "blocked",
            "input_validation_details": details,
            # Pre-populate the refusal so the conditional edge can route
            # straight to END without classify firing.
            "response": INPUT_VALIDATION_REFUSAL,
            "route": "trivial",
            "classifier_raw": "(input-validation-blocked)",
        }

    if action == "normalized":
        LG_INPUT_VALIDATION_TOTAL.labels(action="normalized", reason="").inc()
        log.info("input_validation normalized details=%s", details)
        return {
            "input_validation_action": "normalized",
            "input_validation_details": details,
            "prompt": sanitized,  # downstream nodes see sanitized form
        }

    LG_INPUT_VALIDATION_TOTAL.labels(action="passed", reason="").inc()
    return {
        "input_validation_action": "passed",
        "input_validation_details": details,
    }


def _route_after_input_validation(state: AgentState) -> Literal["safety_input", "__end__"]:
    """Conditional edge: blocked → END; else → safety_input."""
    if state.get("input_validation_action") == "blocked":
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


# --- Semantic prompt cache (Phase #7) --------------------------------------
#
# cache_lookup runs after safety_input. Cache hits skip load_memory,
# rewrite_query, classify, retrieve, ensure_warm, execute, reflect, AND
# safety_output (the cached response was safety-checked when stored).
# They DO go through save_memory so the conversation history reflects
# what the user saw.


def node_cache_lookup(state: AgentState) -> AgentState:
    """Embed prompt, find best match in cache, return hit if above threshold."""
    if not CACHE_ENABLED:
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "cache_lookup_ms": 0,
        }

    user = state.get("user", "unknown")
    session_id = state.get("session_id")
    if not session_id:
        # Cache is per-(user, session). Without session_id we can't isolate.
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "cache_lookup_ms": 0,
        }

    started = time.monotonic()

    # Step 1: embed the prompt. If bge-m3 is unreachable, fail-open
    # (cache miss, downstream pipeline runs normally).
    # Phase #16: prefer redacted_prompt when present so PII doesn't
    # become part of the cache embedding signature. Falls back to
    # raw prompt when input redaction is disabled.
    prompt = state.get("redacted_prompt") or state.get("prompt", "")
    embedding = _embed_prompt_for_cache(prompt)
    if embedding is None:
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "cache_lookup_ms": int((time.monotonic() - started) * 1000),
        }

    redis_client = _get_redis()
    if redis_client is None:
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "prompt_embedding": embedding,
            "cache_lookup_ms": int((time.monotonic() - started) * 1000),
        }

    # Step 2: fetch all entry IDs for this (user, session).
    index_key = _cache_index_key(user, session_id)
    try:
        entry_ids: list[str] = redis_client.zrange(index_key, 0, -1) or []
    except Exception as e:
        log.warning("cache_lookup: zrange failed: %s, fail-open", e)
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "prompt_embedding": embedding,
            "cache_lookup_ms": int((time.monotonic() - started) * 1000),
        }

    if not entry_ids:
        # Empty cache — still pass embedding through so cache_store
        # can use it without re-embedding.
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "prompt_embedding": embedding,
            "cache_lookup_ms": int((time.monotonic() - started) * 1000),
        }

    # Step 3: pipeline-fetch each entry's embedding + response, score
    # cosine similarity in Python. Capped at 20 entries by storage so
    # the linear scan is acceptable.
    import json as _json
    try:
        pipe = redis_client.pipeline(transaction=False)
        for eid in entry_ids:
            pipe.hgetall(_cache_entry_key(user, session_id, eid))
        entries = pipe.execute()
    except Exception as e:
        log.warning("cache_lookup: hgetall pipeline failed: %s, fail-open", e)
        return {
            "cache_hit": False,
            "cache_similarity": 0.0,
            "prompt_embedding": embedding,
            "cache_lookup_ms": int((time.monotonic() - started) * 1000),
        }

    best_score = 0.0
    best_response: Optional[str] = None
    for entry in entries:
        if not entry:
            continue
        emb_json = entry.get("embedding")
        if not emb_json:
            continue
        try:
            cached_emb = _json.loads(emb_json)
        except (_json.JSONDecodeError, TypeError):
            continue
        score = _cosine_similarity(embedding, cached_emb)
        if score > best_score:
            best_score = score
            best_response = entry.get("response", "")

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "cache_lookup user=%s session=%s entries=%d best_score=%.4f threshold=%.4f ms=%d",
        user, session_id, len(entry_ids), best_score,
        CACHE_SIMILARITY_THRESHOLD, elapsed_ms,
    )

    if best_score >= CACHE_SIMILARITY_THRESHOLD and best_response is not None:
        # Cache HIT. Pre-populate response + skip-route fields so the
        # conditional edge after this node routes to save_memory and
        # the user sees the cached answer.
        return {
            "cache_hit": True,
            "cache_similarity": best_score,
            "prompt_embedding": embedding,
            "cache_lookup_ms": elapsed_ms,
            "response": best_response,
            # Skip-route bookkeeping so chat-ui can render a clean
            # "served from cache" badge — classify didn't run, retrieve
            # didn't run, etc.
            "route": "trivial",
            "classifier_raw": "(cache-hit)",
        }

    # Cache MISS — pass embedding through so cache_store can reuse it.
    return {
        "cache_hit": False,
        "cache_similarity": best_score,
        "prompt_embedding": embedding,
        "cache_lookup_ms": elapsed_ms,
    }


def _route_after_cache_lookup(state: AgentState) -> Literal["load_memory", "save_memory"]:
    """Conditional edge: cache HIT → save_memory (skip pipeline). MISS → continue."""
    if state.get("cache_hit"):
        return "save_memory"
    return "load_memory"


def node_cache_store(state: AgentState) -> AgentState:
    """Store the response in cache for future hits.

    Runs after safety_output, before save_memory. Skips on:
      - CACHE_ENABLED false
      - this WAS a cache hit (don't re-store the same entry)
      - safety blocked (don't cache refusals — would short-circuit
        future legitimate similar prompts to refusals)
      - budget blocked (no real response to cache)
      - missing session_id, missing embedding (no key to write under)
    """
    if not CACHE_ENABLED:
        return {"cache_store_ms": 0}
    if state.get("cache_hit"):
        return {"cache_store_ms": 0}
    if state.get("safety_action") in ("blocked_input", "blocked_output"):
        return {"cache_store_ms": 0}
    if state.get("budget_action") in ("exceeded", "fail_closed"):
        return {"cache_store_ms": 0}
    # Phase #9: don't cache responses flagged or blocked as ungrounded
    # — caching a hallucination short-circuits future similar requests
    # to the same wrong answer. "fail_open" passes through (the grader
    # itself failed, not the response).
    if state.get("hallucination_action") in ("flagged", "blocked"):
        return {"cache_store_ms": 0}

    user = state.get("user", "unknown")
    session_id = state.get("session_id")
    if not session_id:
        return {"cache_store_ms": 0}

    embedding = state.get("prompt_embedding")
    if not embedding:
        # cache_lookup didn't compute one (likely bge-m3 was down).
        # Skip rather than re-embed — if it was down at lookup it's
        # probably down now too.
        return {"cache_store_ms": 0}

    response = state.get("response") or ""
    if not response:
        return {"cache_store_ms": 0}

    started = time.monotonic()
    redis_client = _get_redis()
    if redis_client is None:
        return {"cache_store_ms": int((time.monotonic() - started) * 1000)}

    import json as _json
    import uuid as _uuid

    entry_id = _uuid.uuid4().hex[:12]
    entry_key = _cache_entry_key(user, session_id, entry_id)
    index_key = _cache_index_key(user, session_id)
    ts = time.time()
    # Phase #16: cache the redacted prompt, not the raw one. Cache
    # entries persist for CACHE_TTL_SECONDS (24h default); raw PII
    # would otherwise sit in Redis for that whole window.
    prompt = (
        state.get("redacted_prompt")
        or state.get("original_prompt")
        or state.get("prompt")
        or ""
    )

    try:
        # Pipeline: HSET entry, ZADD index, ZREMRANGEBYRANK to keep
        # only N most-recent entries (LRU eviction), then EXPIRE both.
        pipe = redis_client.pipeline(transaction=False)
        pipe.hset(entry_key, mapping={
            "prompt": prompt,
            "embedding": _json.dumps(embedding),
            "response": response,
            "ts": str(ts),
        })
        pipe.expire(entry_key, CACHE_TTL_SECONDS)
        pipe.zadd(index_key, {entry_id: ts})
        # Keep only the most recent CACHE_MAX_ENTRIES_PER_SESSION.
        # ZREMRANGEBYRANK 0 to -(N+1) removes everything except the
        # last N (highest scores = newest).
        pipe.zremrangebyrank(index_key, 0, -(CACHE_MAX_ENTRIES_PER_SESSION + 1))
        pipe.expire(index_key, CACHE_TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        log.warning("cache_store: redis op failed: %s", e)
        return {"cache_store_ms": int((time.monotonic() - started) * 1000)}

    # Note: this leaves orphaned entry HASH keys in Redis after
    # ZREMRANGEBYRANK evicts their index entries. Their TTL cleans
    # them up after CACHE_TTL_SECONDS. A perfect implementation would
    # ZRANGEBYSCORE the evicted IDs and DEL their HASH keys atomically,
    # but the orphan window (24h) is bounded and the storage cost is
    # capped at MAX_ENTRIES * average size ≈ 200KB per session.

    elapsed_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "cache_store user=%s session=%s entry=%s ms=%d",
        user, session_id, entry_id, elapsed_ms,
    )
    return {"cache_store_ms": elapsed_ms}


# --- Runtime hallucination detection (Phase #9) ----------------------------


_HALLUCINATION_SYSTEM_PROMPT = """You grade whether an assistant's answer is grounded in the retrieved context.

You will receive:
- Retrieved context chunks (numbered [1], [2], etc.)
- The assistant's answer

Output exactly ONE LINE of valid JSON:
  {"verdict": "grounded" | "partial" | "ungrounded", "confidence": <0.0-1.0>}

Rules:
- "grounded"   → every substantive claim in the answer is directly supported by at least one chunk, OR is a reasonable inference from one or more chunks.
- "partial"    → some claims supported, others extrapolated beyond what the chunks say.
- "ungrounded" → the answer's substantive claims are NOT in any chunk; the assistant fabricated or used its own training data.

Ignore boilerplate like "I'd be happy to help" or "Let me explain" when grading. Focus on factual claims.

confidence reflects how sure YOU are of YOUR verdict, not the answer's certainty. 0.95 = very sure. 0.5 = could go either way.

Output ONLY the JSON line. No prose, no code fences, no explanation."""


def _hallucination_grade(
    response: str,
    chunks: list[dict],
) -> tuple[str, float, int]:
    """Call Llama 8B to grade whether response is grounded in chunks.

    Returns (verdict, confidence, latency_ms). Verdict is one of
    "grounded" | "partial" | "ungrounded" | "fail_open" (LLM error or
    parse failure → fail-open verdict treated as passed by caller).
    """
    started = time.monotonic()

    # Format the chunks the same shape node_execute already uses,
    # so the grader sees citation indexes consistent with what the
    # original answer was generated against.
    context_text = "\n\n".join(
        "[{n}] (source: {src}, chunk {ci})\n{text}".format(
            n=i + 1,
            src=c.get("source", "unknown") or "unknown",
            ci=c.get("chunk_index", 0),
            text=(c.get("text", "") or "")[:1500],  # cap each chunk
        )
        for i, c in enumerate(chunks[:5])  # cap at 5 chunks for prompt budget
    )

    user_msg = f"Retrieved context:\n{context_text}\n\nAssistant answer:\n{response}\n\nVerdict JSON:"

    cfg_trivial = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg_trivial["model_name"],
        base_url=cfg_trivial["url"],
        api_key="not-required",
        temperature=0.0,
        max_tokens=HALLUCINATION_MAX_TOKENS,
        timeout=HALLUCINATION_TIMEOUT_SECONDS,
    )
    try:
        resp = client.invoke([
            SystemMessage(content=_HALLUCINATION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = (resp.content or "").strip()
    except Exception as e:
        log.warning("hallucination_grade: LLM call failed: %s, fail-open", e)
        return ("fail_open", 0.0, int((time.monotonic() - started) * 1000))

    # Parse the first JSON object from the response. Llama 8B
    # occasionally wraps in code fences or adds trailing whitespace.
    import json as _json
    import re as _re
    m = _re.search(r"\{[^{}]*\}", raw, _re.DOTALL)
    if not m:
        log.warning("hallucination_grade: no JSON in output=%r", raw[:120])
        return ("fail_open", 0.0, int((time.monotonic() - started) * 1000))
    try:
        decision = _json.loads(m.group(0))
    except _json.JSONDecodeError:
        log.warning("hallucination_grade: malformed JSON=%r", m.group(0)[:120])
        return ("fail_open", 0.0, int((time.monotonic() - started) * 1000))

    verdict = (decision.get("verdict") or "").lower().strip()
    if verdict not in ("grounded", "partial", "ungrounded"):
        log.warning("hallucination_grade: unknown verdict=%r", verdict)
        return ("fail_open", 0.0, int((time.monotonic() - started) * 1000))

    try:
        confidence = float(decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Apply the confidence threshold: a "grounded" verdict with low
    # confidence gets demoted to "partial" so HALLUCINATION_ACTION can
    # treat low-confidence-grounded the same as partial.
    if verdict == "grounded" and confidence < HALLUCINATION_CONFIDENCE_THRESHOLD:
        log.info(
            "hallucination_grade: low-confidence grounded (%.2f < %.2f), demoting to partial",
            confidence, HALLUCINATION_CONFIDENCE_THRESHOLD,
        )
        verdict = "partial"

    return (verdict, confidence, elapsed_ms)


def node_hallucination_check(state: AgentState) -> AgentState:
    """Grade response groundedness; flag or block per HALLUCINATION_ACTION."""
    if not HALLUCINATION_CHECK_ENABLED:
        return {
            "hallucination_verdict": "skipped",
            "hallucination_confidence": 0.0,
            "hallucination_action": "disabled",
            "hallucination_check_ms": 0,
        }

    # Skip on cache hit — already-checked content from cache_store.
    if state.get("cache_hit"):
        return {
            "hallucination_verdict": "skipped",
            "hallucination_confidence": 0.0,
            "hallucination_action": "passed",
            "hallucination_check_ms": 0,
        }

    # Skip if safety blocked — response IS the refusal text, not
    # subject to grounding check.
    if state.get("safety_action") in ("blocked_input", "blocked_output"):
        return {
            "hallucination_verdict": "skipped",
            "hallucination_confidence": 0.0,
            "hallucination_action": "passed",
            "hallucination_check_ms": 0,
        }

    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        # No retrieval context — model used its own knowledge, no
        # grounding check applies. Pass through.
        return {
            "hallucination_verdict": "skipped",
            "hallucination_confidence": 0.0,
            "hallucination_action": "passed",
            "hallucination_check_ms": 0,
        }

    response = state.get("response", "") or ""
    if not response:
        return {
            "hallucination_verdict": "skipped",
            "hallucination_confidence": 0.0,
            "hallucination_action": "passed",
            "hallucination_check_ms": 0,
        }

    verdict, confidence, ms = _hallucination_grade(response, chunks)
    log.info(
        "hallucination_check verdict=%s confidence=%.2f ms=%d action_mode=%s",
        verdict, confidence, ms, HALLUCINATION_ACTION,
    )

    # fail_open is treated as "passed" — the check failed, but we
    # don't want a flaky LLM call to disable the feature. Surfaces
    # in telemetry so operators can audit how often it fires.
    if verdict in ("grounded", "fail_open"):
        return {
            "hallucination_verdict": verdict,
            "hallucination_confidence": confidence,
            "hallucination_action": "passed",
            "hallucination_check_ms": ms,
        }

    # verdict in ("partial", "ungrounded") — apply HALLUCINATION_ACTION
    if HALLUCINATION_ACTION == "block":
        new_response = HALLUCINATION_DISCLAIMER + response
        return {
            "hallucination_verdict": verdict,
            "hallucination_confidence": confidence,
            "hallucination_action": "blocked",
            "hallucination_check_ms": ms,
            "response": new_response,
        }
    # default: "flag" — record verdict, leave response alone
    return {
        "hallucination_verdict": verdict,
        "hallucination_confidence": confidence,
        "hallucination_action": "flagged",
        "hallucination_check_ms": ms,
    }


# --- PII redaction at input (Phase #16) ------------------------------------
#
# Detects PII in the user's prompt and stores a redacted copy in
# state.redacted_prompt. Downstream nodes (cache_lookup, retrieve,
# rewrite_query, save_memory) read state.redacted_prompt instead of
# state.prompt so PII never lands in cache keys, vector-search queries,
# or conversation history. node_execute still uses state.prompt — the
# model needs the user's actual content to answer.
#
# Position: AFTER safety_input (Llama Guard scans the original — its
# safety verdict needs the actual content), BEFORE cache_lookup (so
# cache embedding is on the redacted form, preventing PII-keyed cache
# entries) and load_memory.
#
# Reuses the regex detector + redactor from Phase #11.


def node_pii_redact_input(state: AgentState) -> AgentState:
    """Detect PII in state.prompt; populate state.redacted_prompt.

    When disabled OR no PII found, redacted_prompt is identical to
    prompt (downstream nodes read it the same way regardless of
    whether redaction actually changed anything).
    """
    prompt = state.get("prompt", "") or ""
    if not PII_REDACT_INPUT_ENABLED or not prompt:
        return {
            "redacted_prompt": prompt,
            "pii_input_action": "skipped",
            "pii_input_entities_found": {},
            "pii_input_redact_ms": 0,
        }

    started = time.monotonic()
    spans = _detect_pii(prompt)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not spans:
        return {
            "redacted_prompt": prompt,
            "pii_input_action": "passed",
            "pii_input_entities_found": {},
            "pii_input_redact_ms": elapsed_ms,
        }

    counts: dict[str, int] = {}
    for entity_type, _, _ in spans:
        counts[entity_type] = counts.get(entity_type, 0) + 1

    redacted = _redact_pii(prompt, spans)
    log.info(
        "pii_redact_input entities=%s ms=%d",
        counts, elapsed_ms,
    )
    return {
        "redacted_prompt": redacted,
        "pii_input_action": "redacted",
        "pii_input_entities_found": counts,
        "pii_input_redact_ms": elapsed_ms,
    }


# --- PII redaction at output (Phase #11) -----------------------------------


def node_pii_redact_output(state: AgentState) -> AgentState:
    """Scan response for PII and replace with <redacted_TYPE> placeholders."""
    if not PII_REDACT_OUTPUT_ENABLED:
        return {
            "pii_redact_action": "skipped",
            "pii_entities_found": {},
            "pii_redact_ms": 0,
        }

    # Skip on cache hit — already-redacted from cache_store.
    if state.get("cache_hit"):
        return {
            "pii_redact_action": "skipped",
            "pii_entities_found": {},
            "pii_redact_ms": 0,
        }

    # Skip if safety blocked — response is the refusal text, not subject
    # to redaction (and shouldn't contain PII anyway).
    if state.get("safety_action") in ("blocked_input", "blocked_output"):
        return {
            "pii_redact_action": "skipped",
            "pii_entities_found": {},
            "pii_redact_ms": 0,
        }

    response = state.get("response", "") or ""
    if not response:
        return {
            "pii_redact_action": "skipped",
            "pii_entities_found": {},
            "pii_redact_ms": 0,
        }

    started = time.monotonic()
    spans = _detect_pii(response)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not spans:
        return {
            "pii_redact_action": "passed",
            "pii_entities_found": {},
            "pii_redact_ms": elapsed_ms,
        }

    # Count by entity type — NEVER expose the original values.
    counts: dict[str, int] = {}
    for entity_type, _, _ in spans:
        counts[entity_type] = counts.get(entity_type, 0) + 1

    redacted = _redact_pii(response, spans)
    log.info(
        "pii_redact_output entities=%s ms=%d",
        counts, elapsed_ms,
    )
    return {
        "pii_redact_action": "redacted",
        "pii_entities_found": counts,
        "pii_redact_ms": elapsed_ms,
        "response": redacted,
    }


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
    # Phase #16: rewrite operates on the redacted prompt so the
    # refined_query produced (used for retrieval, possibly logged
    # downstream) doesn't contain PII. The "original_prompt" state
    # field name predates redaction — kept for API compatibility but
    # really means "input to rewrite" which is now the redacted form
    # when redaction is on.
    original_prompt = state.get("redacted_prompt") or state.get("prompt", "")
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
        # Phase #16: persist the redacted prompt to memory so PII
        # doesn't sit in conversation history. Falls back to
        # original_prompt (the rewrite-input) and finally to raw
        # prompt when redaction is off.
        "prompt": (
            state.get("redacted_prompt")
            or state.get("original_prompt")
            or state.get("prompt")
            or ""
        ),
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
    # prompt. After this consumes it, we reset it to None so the
    # next reflect call doesn't see stale data.
    #
    # Phase #16: prefer redacted_prompt over raw prompt when present so
    # PII doesn't end up in retrieval queries (which may be logged by
    # rag-service or its observability stack). refined_query already
    # passes through pii-aware contexts (rewrite operates on redacted),
    # so the falback chain is: refined_query → redacted_prompt → prompt.
    query_for_retrieval = (
        state.get("refined_query")
        or state.get("redacted_prompt")
        or state["prompt"]
    )
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

# --- Phase #13: plan-and-execute agent pattern -----------------------------
#
# The current agentic loop in node_execute is ReAct-style — model sees a
# question + tools, decides one tool to call, sees result, decides next
# tool, iterates. Effective for simple multi-step tasks but degrades on
# complex tasks where the model loses track of what it's done vs what
# remains.
#
# Plan-and-execute adds an UPFRONT planning step: before any tool calls,
# the model produces a structured plan (numbered list of steps), then
# the execute loop is gently steered by that plan via an additional
# system message. Empirically improves multi-step task completion vs
# bare ReAct (LangChain blog, 2023; Plan-and-Solve paper).
#
# Per-tier opt-in via ROUTE_REGISTRY[<tier>]["use_planner"] — planning
# helps most on hard/reasoning tiers where the prompt is multi-faceted.
# trivial-tier prompts ("what's 2+2") shouldn't pay the ~500ms planning
# overhead.
#
# Default disabled globally (PLANNER_ENABLED=false) so this commit lands
# without changing observable behavior.

PLANNER_ENABLED = os.environ.get("PLANNER_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
PLANNER_MAX_STEPS = int(os.environ.get("PLANNER_MAX_STEPS", "5"))
PLANNER_MAX_TOKENS = int(os.environ.get("PLANNER_MAX_TOKENS", "256"))
PLANNER_TIMEOUT_SECONDS = float(os.environ.get("PLANNER_TIMEOUT_SECONDS", "20"))

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

# --- Phase #20: content-based input validation -----------------------------
#
# Cheap regex/string defense layer for attack classes Llama Guard isn't
# specifically trained on:
#
#   homoglyph attack    "ⅈgnore previous instructions" with a Cyrillic
#                       lookalike for a Latin letter. Normalizing to
#                       NFKC collapses confusables to canonical form
#                       so safety filters see the actual instruction.
#   zero-width smuggle  "act​normal​ but actually..." —
#                       hidden characters split tokens to evade
#                       keyword detection. Strip them.
#   context exhaustion  10K+ char prompts to crowd out the system
#                       prompt or balloon per-token rate limits.
#                       Hard-cap length.
#   control char inject U+0000 / U+0007 / etc. Some downstream
#                       parsers / loggers / terminals choke. Block.
#   whitespace padding  10K+ space chars to defeat per-byte rate
#                       limits without making the model do real work.
#                       Block.
#
# Position: AFTER budget_check (cheapest), BEFORE safety_input (Llama
# Guard at ~300ms shouldn't be invoked on malformed input).
#
# Two-mode behavior:
#   normalize  unicode + zero-width — sanitize in-place, update
#              state.prompt for downstream nodes
#   block      length / control / excessive-whitespace — short-circuit
#              to END with INPUT_VALIDATION_REFUSAL
#
# Default disabled.

VALIDATE_INPUT_ENABLED = os.environ.get(
    "VALIDATE_INPUT_ENABLED", "false"
).lower() in ("1", "true", "yes")
# 16K chars covers any legitimate technical question + retrieved
# context. Anything longer is almost certainly an attack OR a user
# pasting an entire log dump (legitimate but better handled via
# explicit upload, not /invoke).
VALIDATE_INPUT_MAX_LENGTH = int(
    os.environ.get("VALIDATE_INPUT_MAX_LENGTH", "16384")
)
INPUT_VALIDATION_REFUSAL = os.environ.get(
    "INPUT_VALIDATION_REFUSAL",
    "I can't process this request as-is. Please try a shorter or differently formatted prompt.",
)


_ZERO_WIDTH_CHARS = "​‌‍﻿⁠"


def _validate_input(text: str) -> tuple[Optional[str], str, dict]:
    """Validate + sanitize user input.

    Returns (sanitized_or_None, action, details):
      sanitized=None → caller blocks the request
      sanitized=str → caller updates state.prompt to this
      action: "passed" | "normalized" | "blocked"
      details: per-check counts / reasons (NEVER raw matched content)
    """
    details: dict = {}

    # Hard length cap first — cheapest check, cuts off whitespace
    # padding attacks before the regex scan.
    if len(text) > VALIDATE_INPUT_MAX_LENGTH:
        return (None, "blocked", {
            "reason": "length_exceeded",
            "length": len(text),
            "max": VALIDATE_INPUT_MAX_LENGTH,
        })

    # Control chars — anything in C0/C1 except the common whitespace
    # the model legitimately handles (\t \n \r). Excludes 0x7F (DEL)
    # which is also a common injection attempt.
    control_count = sum(
        1
        for c in text
        if (ord(c) < 0x20 and c not in "\t\n\r") or ord(c) == 0x7F
    )
    if control_count > 0:
        return (None, "blocked", {
            "reason": "control_chars",
            "count": control_count,
        })

    # Excessive whitespace runs — defeat per-byte cost models without
    # making the model do real work. 1000 consecutive whitespace chars
    # is generous (legitimate text rarely has more than 4-5 in a row).
    import re as _re
    long_ws = _re.search(r"\s{1000,}", text)
    if long_ws is not None:
        return (None, "blocked", {
            "reason": "excessive_whitespace",
            "length": long_ws.end() - long_ws.start(),
        })

    # Sanitization (in-place fixes; doesn't block):
    # Strip zero-width chars first.
    sanitized = text
    zw_count = sum(sanitized.count(c) for c in _ZERO_WIDTH_CHARS)
    if zw_count:
        for c in _ZERO_WIDTH_CHARS:
            sanitized = sanitized.replace(c, "")
        details["zero_width_stripped"] = zw_count

    # Then NFKC normalize. NFKC > NFC because it collapses
    # compatibility lookalikes (mathematical bold "𝐞" → "e", Cyrillic
    # 'е' → 'e' for the lookalike subset, fullwidth → ASCII). Stops
    # most homoglyph variants without killing legitimate Unicode
    # like emoji.
    import unicodedata as _ud
    normalized = _ud.normalize("NFKC", sanitized)
    if normalized != sanitized:
        details["unicode_normalized"] = True
        sanitized = normalized

    if sanitized != text:
        return (sanitized, "normalized", details)
    return (text, "passed", details)

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


# --- Phase #7: semantic prompt cache --------------------------------------
#
# Embedding-similarity-based prompt cache. On a hit, the entire downstream
# pipeline (load_memory, rewrite_query, classify, retrieve, ensure_warm,
# execute, reflect, safety_output) is skipped and the cached response is
# returned. Halves cost on workloads with repetitive queries (typical chat
# workload sees 30-50% repetition rate).
#
# Position in graph: AFTER safety_input (so unsafe prompts don't bypass
# safety even if cached), BEFORE load_memory (so we save load+rewrite cost
# on hits). Trade-off worth knowing: the cache key is the RAW prompt
# embedding, not the rewritten one — meaning two semantically-equivalent
# follow-up questions with different pronouns ("how does it work?" vs
# "how does Pod Identity work?") may NOT cache-hit each other. Future
# v2 could move cache_lookup AFTER rewrite_query for better hit rate at
# the cost of always paying the rewrite latency.
#
# Storage layout in langgraph-redis:
#   cache:<user>:<session>:index    Redis SORTED SET. entry_id → ts.
#                                   ZREMRANGEBYRANK at write time keeps
#                                   only CACHE_MAX_ENTRIES_PER_SESSION
#                                   most-recent entries (LRU eviction).
#   cache:<user>:<session>:<id>     Redis HASH with fields:
#                                     prompt (str), embedding (json
#                                     array of 1024 floats), response
#                                     (str), ts (float).
#
# Both keys carry CACHE_TTL_SECONDS (default 24h) — cache loses freshness
# faster than memory because retrieval results may have changed (new
# documents uploaded, etc.). Active sessions refresh TTL on every store.
#
# Failure modes (all fail-OPEN — the request still completes the full
# pipeline if cache infrastructure is degraded):
#   - vllm-bge-m3 unreachable → no embedding → cache miss
#   - Redis unreachable → cache miss
#   - Embedding malformed → cache miss
#
# Default disabled. Activation requires vllm-bge-m3 to be Ready (the
# embedding model used to compute similarity).

CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
CACHE_EMBEDDINGS_URL = os.environ.get(
    "CACHE_EMBEDDINGS_URL", "http://vllm-bge-m3.llm.svc.cluster.local:8000/v1"
)
CACHE_EMBEDDINGS_MODEL = os.environ.get("CACHE_EMBEDDINGS_MODEL", "bge-m3")
# Cosine similarity threshold for a hit. 0.95 is "very similar but not
# identical" — typical for paraphrased questions about the same topic.
# Tighten (e.g. 0.98) if the cache returns false-positive answers from
# different-but-related prompts; loosen (e.g. 0.90) if you want broader
# hits at the cost of occasional unrelated cached responses.
CACHE_SIMILARITY_THRESHOLD = float(
    os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.95")
)
# Per-(user, session) entry cap. 20 entries × ~10KB per entry (prompt +
# 1024 floats as JSON + response + metadata) = ~200KB per session in
# Redis. Negligible at lab scale, manageable at production scale.
CACHE_MAX_ENTRIES_PER_SESSION = int(
    os.environ.get("CACHE_MAX_ENTRIES_PER_SESSION", "20")
)
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", str(24 * 3600)))
CACHE_TIMEOUT_SECONDS = float(os.environ.get("CACHE_TIMEOUT_SECONDS", "5"))


# --- Phase #9: runtime hallucination detection ----------------------------
#
# After the model produces its draft answer, ask Llama 8B to grade
# whether the answer is grounded in the retrieved chunks. Three outcomes:
#
#   "grounded"    answer's claims appear in (or are reasonable inferences
#                 from) the retrieved chunks. Pass through unchanged.
#   "partial"     some claims grounded, others extrapolated. Flag in
#                 telemetry; behavior depends on HALLUCINATION_ACTION.
#   "ungrounded"  answer's claims aren't in the retrieved chunks at all
#                 (model fell back to its training data, or fabricated).
#                 Action depends on HALLUCINATION_ACTION:
#                   "flag"  — record verdict, return response unchanged
#                   "block" — prepend a disclaimer like "I'm not
#                             confident in this answer because the
#                             retrieved sources didn't support it."
#
# Position: AFTER safety_output (so we don't grade refusal text against
# retrieved chunks — refusals aren't trying to be grounded), BEFORE
# cache_store (so we don't cache responses we flagged as ungrounded —
# would short-circuit future similar requests to the same hallucinated
# answer).
#
# Skip cases:
#   - HALLUCINATION_CHECK_ENABLED=false
#   - cache_hit (cached response was checked when stored)
#   - safety blocked (response IS the refusal text, not subject to
#     grounding check)
#   - no retrieved chunks (the model legitimately used its own
#     knowledge — there's no context to ground against)
#
# Cost: one Llama 8B call per request. ~3000-token input (chunks +
# response), ~50-token output, ~500ms-1s warm. Adds noticeable latency
# vs the no-check path. Production would consider running the check
# async (return response now, surface verdict in a follow-up event)
# but for the lab a synchronous check keeps the API simple.

HALLUCINATION_CHECK_ENABLED = os.environ.get("HALLUCINATION_CHECK_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
# "flag" or "block". flag = record verdict, leave response alone. block =
# prepend a confidence disclaimer to the response on ungrounded verdict.
HALLUCINATION_ACTION = os.environ.get("HALLUCINATION_ACTION", "flag").lower()
HALLUCINATION_TIMEOUT_SECONDS = float(
    os.environ.get("HALLUCINATION_TIMEOUT_SECONDS", "20")
)
HALLUCINATION_MAX_TOKENS = int(os.environ.get("HALLUCINATION_MAX_TOKENS", "128"))
# Confidence threshold below which a "grounded" verdict is treated as
# "partial." Llama 8B at temp=0 still has variance in its confidence
# language; thresholding lets operators dial sensitivity. 0.6 = "the
# model is more confident than not it's grounded."
HALLUCINATION_CONFIDENCE_THRESHOLD = float(
    os.environ.get("HALLUCINATION_CONFIDENCE_THRESHOLD", "0.6")
)
# Disclaimer text prepended to the response when verdict is ungrounded
# AND HALLUCINATION_ACTION=block. Phrased as a hedging note rather than
# a refusal so users still see the model's draft and can judge it.
HALLUCINATION_DISCLAIMER = os.environ.get(
    "HALLUCINATION_DISCLAIMER",
    "[Note: my answer below may not be fully supported by the retrieved sources. Please verify.]\n\n",
)


# --- Phase #10: user feedback loop -----------------------------------------
#
# Companion to /invoke + /invoke/stream: a separate POST /feedback endpoint
# where chat-ui (or any client) submits per-response ratings. This is the
# foundation for the data flywheel — without user signal, you can't do
# online eval, A/B testing, RLHF data collection, or continuous fine-tune.
# Builds the cheapest piece first (capture); analytics/feedback-driven
# training are downstream concerns.
#
# Storage: dual-write to Redis (operational) + Langfuse score API
# (visibility in the trace UI).
#   Redis: hash keyed feedback:<user>:<trace_id> with rating + comment +
#     categories + ts. TTL = FEEDBACK_TTL_SECONDS (default 90d). Indexed
#     by a per-user list feedback:<user>:list for O(1) recent-feedback
#     lookups.
#   Langfuse: a `score` is an attribute attached to a trace by ID. The
#     Langfuse UI then shows feedback inline with each conversation,
#     and aggregate scoring is queryable via the SDK or web UI.
#
# Auth: same Keycloak JWT as /invoke. The feedback is keyed by
# (user, trace_id) so users can only submit feedback on their own
# requests — checked by binding the username at submission time.
#
# Failure modes:
#   - Redis unreachable → 503 with detail. No silent failure; feedback
#     loss is a quality-bar regression and operators should know.
#   - Langfuse unreachable → log + continue. Redis is the persistent
#     store; Langfuse is the dashboard view, so failing partly is
#     acceptable.

FEEDBACK_TTL_SECONDS = int(os.environ.get("FEEDBACK_TTL_SECONDS", str(90 * 86400)))
# Cap on per-user feedback list — keeps the index ZRANGE cheap.
# Older entries are still in their hash keys (TTL'd separately) but
# the index only points at the most-recent N.
FEEDBACK_INDEX_MAX_ENTRIES = int(
    os.environ.get("FEEDBACK_INDEX_MAX_ENTRIES", "500")
)


def _feedback_key(user: str, trace_id: str) -> str:
    return f"feedback:{user}:{trace_id}"


def _feedback_index_key(user: str) -> str:
    return f"feedback:{user}:list"


# Lazy-initialized Langfuse client for the programmatic score API.
# Distinct from _LANGFUSE_CB which is a LangChain callback handler;
# the score API is a separate SDK surface. None when LANGFUSE_*
# env vars are unset (lab without trace export).
_LANGFUSE_CLIENT = None


def _get_langfuse_client():
    """Return a Langfuse SDK client instance, or None if not configured."""
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return None
    try:
        from langfuse import Langfuse as _Langfuse
        _LANGFUSE_CLIENT = _Langfuse()
        return _LANGFUSE_CLIENT
    except Exception as e:
        log.warning("langfuse client init failed: %s", e)
        return None


# --- Phase #11: PII redaction at output ------------------------------------
#
# Scan the model's response for common PII patterns and mask them before
# the response leaves the graph. Catches the failure mode "model
# regurgitated PII from training data or retrieved chunks" — both are
# real risks in RAG systems where the corpus might contain customer
# data, AWS credentials, internal IPs, etc.
#
# Position: AFTER hallucination_check (the grader needs the unredacted
# response to compare claims against chunks; redaction would lose info),
# BEFORE cache_store (don't cache PII-containing responses) and BEFORE
# save_memory (don't store PII in conversation history).
#
# Detection: regex-based. Six entity types supported in v1:
#   email, phone_us, ssn, credit_card, ipv4, aws_access_key
# Lightweight, no new dependency. For higher-fidelity detection
# (Microsoft Presidio + spaCy NER for names, addresses, organizations),
# swap in a future Phase #11.5 — same node interface.
#
# Replacement: matched spans replaced with <redacted_<TYPE>> in-place
# (e.g. "email me at <redacted_email>"). Preserves the surrounding
# sentence structure so the response remains readable.
#
# Telemetry: surfaces COUNTS by entity type in InvokeResponse — never
# the original values (would defeat the redaction by leaking via API).
#
# Future work this enables (NOT in this commit):
#   - Input-side redaction (don't put PII in retrieval queries, cache
#     keys, or memory). Requires refactoring cache_lookup/save_memory
#     to use a redacted_prompt field. Larger scope.
#   - PII-aware policy: if too many PII entities found, refuse to
#     answer (treat similar to a safety block).

PII_REDACT_OUTPUT_ENABLED = os.environ.get(
    "PII_REDACT_OUTPUT_ENABLED", "false"
).lower() in ("1", "true", "yes")

# Phase #16: input-side PII redaction. Detects PII in the user's prompt
# and produces a redacted version used for cache keys, retrieval search
# queries, memory storage, and rewrite-query input. The ORIGINAL prompt
# still goes to node_execute (the model needs the user's actual question).
#
# This closes the PII compliance picture: input redaction prevents PII
# from being indexed in cache/memory/retrieval logs; output redaction
# (Phase #11) prevents PII from leaking back through the response.
# Together they bound where PII can sit in the system to a single hot
# path: prompt → execute → response, never touching durable storage.
PII_REDACT_INPUT_ENABLED = os.environ.get(
    "PII_REDACT_INPUT_ENABLED", "false"
).lower() in ("1", "true", "yes")

# Entity types to redact, comma-separated. Default = all six. Operators
# can narrow (e.g. "ipv4,aws_access_key") for a use case where natural
# PII like emails is expected and shouldn't be masked.
PII_REDACT_ENTITY_TYPES = set(
    s.strip().lower()
    for s in os.environ.get(
        "PII_REDACT_ENTITY_TYPES",
        "email,phone_us,ssn,credit_card,ipv4,aws_access_key",
    ).split(",")
    if s.strip()
)


# Regex patterns for each entity type. Compiled once at module load
# for hot-path performance — node_pii_redact_output runs per request,
# and re-compiling on every call wastes CPU. Patterns are deliberately
# conservative (favor false negatives over false positives) since
# over-aggressive redaction of an unrelated string would corrupt
# legitimate responses.
import re as _re

_PII_PATTERNS: dict[str, "_re.Pattern"] = {
    # Standard email format. \b boundaries prevent matching inside
    # longer alphanumeric runs.
    "email": _re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    # US-style phone numbers in common formats: 555-555-5555,
    # (555) 555-5555, +1 555 555 5555, 555.555.5555. Doesn't match
    # bare 10-digit numbers (too prone to matching ZIP+phone codes
    # or any 10-digit identifier).
    "phone_us": _re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    # US SSN format: ddd-dd-dddd. Excludes 000-, 666-, and 9xx- per
    # SSA invalid prefix rules to slightly reduce false positives,
    # but still matches the canonical shape.
    "ssn": _re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    # Credit card: 13-19 digits with optional dashes/spaces in 4-digit
    # groups. NO Luhn check in v1 — would catch more cases at small
    # additional cost. Future enhancement.
    "credit_card": _re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b"
    ),
    # IPv4 dotted quad. Doesn't validate octet ranges (matches
    # 999.999.999.999) — false positives cost less than missing
    # internal IPs leaking in responses.
    "ipv4": _re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # AWS access key ID. Distinct AKIA/ASIA/AGPA prefix + exactly 16
    # uppercase alphanumerics. Very specific shape; near-zero false
    # positive rate. Secret access keys aren't matched (too generic
    # — high-entropy 40-char base64 strings are too prone to false
    # positives in normal text).
    "aws_access_key": _re.compile(r"\b(?:AKIA|ASIA|AGPA|AROA)[0-9A-Z]{16}\b"),
}


def _detect_pii(text: str) -> list[tuple[str, int, int]]:
    """Find PII spans in text. Returns [(entity_type, start, end), ...]
    sorted by end-position descending so callers can replace right-to-left
    without breaking earlier offsets."""
    if not text:
        return []
    spans: list[tuple[str, int, int]] = []
    for entity_type, pattern in _PII_PATTERNS.items():
        if entity_type not in PII_REDACT_ENTITY_TYPES:
            continue
        for match in pattern.finditer(text):
            spans.append((entity_type, match.start(), match.end()))
    # Right-to-left so .replace by offset doesn't shift later spans
    spans.sort(key=lambda s: s[1], reverse=True)
    return spans


def _redact_pii(text: str, spans: list[tuple[str, int, int]]) -> str:
    """Replace each span with <redacted_TYPE>. Spans must be in
    right-to-left order (as returned by _detect_pii) so offsets stay
    valid as we modify the string."""
    out = text
    for entity_type, start, end in spans:
        out = out[:start] + f"<redacted_{entity_type}>" + out[end:]
    return out


def _cache_index_key(user: str, session_id: str) -> str:
    return f"cache:{user}:{session_id}:index"


def _cache_entry_key(user: str, session_id: str, entry_id: str) -> str:
    return f"cache:{user}:{session_id}:{entry_id}"


def _embed_prompt_for_cache(prompt: str) -> Optional[list[float]]:
    """Call vllm-bge-m3 /v1/embeddings. Returns None on failure (fail-open)."""
    try:
        with httpx.Client(timeout=CACHE_TIMEOUT_SECONDS) as client:
            resp = client.post(
                f"{CACHE_EMBEDDINGS_URL}/embeddings",
                json={"model": CACHE_EMBEDDINGS_MODEL, "input": prompt},
            )
            resp.raise_for_status()
            body = resp.json()
            return body["data"][0]["embedding"]
    except Exception as e:
        log.warning("cache: embedding failed: %s, fail-open", e)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain Python cosine similarity. Avoids numpy as a hard dep —
    1024-dim × 20 entries is ~60K multiplies per cache lookup, ~6ms at
    Python speeds. Acceptable for the cache hot path."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))

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


# --- Phase #19: per-tool rate limiting -------------------------------------
#
# Per-(tool, user) sliding-window rate limit. The Phase #5 budget caps
# total /invoke requests per user per day. Per-tool limits cap how many
# times each user can invoke each specific tool in a sliding window.
#
# Why both: the per-user budget catches blanket abuse ("user is asking
# 10K questions"). Per-tool limits catch tool-specific abuse ("user is
# asking benign questions but each one tool-calls http_fetch 50 times,
# scraping the web through us"). Different threat models, different
# mitigations.
#
# Default per-tool limits (calls/window):
#   calculator              60/min  (cheap, abuse-resistant — pure math)
#   get_current_time        60/min  (cheap, no side-effects)
#   http_fetch              10/min  (network egress; scrape risk; SSRF
#                                   risk if URL allowlist were ever
#                                   loosened beyond https-only)
#   search_session_docs     30/min  (embedding cost + Qdrant load)
#
# Override via env: TOOL_RATE_<TOOL_NAME_UPPER>_LIMIT and
# TOOL_RATE_<TOOL_NAME_UPPER>_WINDOW_SECONDS.
#
# Behavior on rate-limit hit: the dispatched call returns a synthetic
# ToolMessage with text "tool <name> rate-limited; retry in <X>s" —
# the agent loop sees it like any other tool error and can decide to
# retry, fall back to a different tool, or give up gracefully. Better
# UX than a hard request failure.
#
# Failure mode (Redis unreachable): fail-OPEN. The whole feature is
# enrichment, not a security boundary; if Redis is down the request
# proceeds normally. Same posture as Phase #5 budget when configured
# fail_open.

TOOL_RATE_LIMIT_ENABLED = os.environ.get(
    "TOOL_RATE_LIMIT_ENABLED", "false"
).lower() in ("1", "true", "yes")

# Defaults populated at module load. Env can override per-tool via
# TOOL_RATE_<TOOL>_LIMIT / TOOL_RATE_<TOOL>_WINDOW_SECONDS.
TOOL_RATE_DEFAULTS: dict[str, dict] = {
    "calculator":          {"limit": 60, "window_seconds": 60},
    "get_current_time":    {"limit": 60, "window_seconds": 60},
    "http_fetch":          {"limit": 10, "window_seconds": 60},
    "search_session_docs": {"limit": 30, "window_seconds": 60},
}

TOOL_RATE_CONFIG: dict[str, dict] = {}
for _tool_name, _default in TOOL_RATE_DEFAULTS.items():
    _env_prefix = f"TOOL_RATE_{_tool_name.upper()}"
    _limit_raw = os.environ.get(f"{_env_prefix}_LIMIT", "").strip()
    _window_raw = os.environ.get(f"{_env_prefix}_WINDOW_SECONDS", "").strip()
    try:
        _limit = int(_limit_raw) if _limit_raw else _default["limit"]
    except ValueError:
        _limit = _default["limit"]
    try:
        _window = int(_window_raw) if _window_raw else _default["window_seconds"]
    except ValueError:
        _window = _default["window_seconds"]
    TOOL_RATE_CONFIG[_tool_name] = {
        "limit": max(1, _limit),
        "window_seconds": max(1, _window),
    }


def _check_tool_rate_limit(tool_name: str, user: str) -> tuple[bool, int]:
    """Return (allowed, remaining). remaining=-1 when feature disabled
    or fail-open.

    Sliding-window via Redis: bucket key includes
    `int(time.time()) // window_seconds` so each window gets its own
    counter that auto-expires. Same INCR + EXPIRE pattern as Phase #5
    budget.
    """
    if not TOOL_RATE_LIMIT_ENABLED:
        return (True, -1)
    cfg = TOOL_RATE_CONFIG.get(tool_name)
    if not cfg:
        # Tool isn't in the registry — let it through. Adding new
        # tools to TOOLS without a matching TOOL_RATE_DEFAULTS entry
        # falls open rather than randomly denying.
        return (True, -1)

    redis_client = _get_redis()
    if redis_client is None:
        return (True, -1)

    window_id = int(time.time()) // cfg["window_seconds"]
    key = f"toolrate:{tool_name}:{user}:{window_id}"
    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.incr(key, 1)
        # 2x window to be safe against clock skew between this pod
        # and Redis on a window boundary
        pipe.expire(key, cfg["window_seconds"] * 2)
        results = pipe.execute()
        consumed = int(results[0])
    except Exception as e:
        log.warning("tool_rate_limit: redis op failed: %s, fail-open", e)
        return (True, -1)

    remaining = cfg["limit"] - consumed
    if consumed > cfg["limit"]:
        return (False, 0)
    return (True, max(0, remaining))


# --- Planner (Phase #13) ---------------------------------------------------


_PLANNER_SYSTEM_PROMPT = """You are a planning agent. Given the user's request, retrieved context, and available tools, produce a SHORT structured plan as a numbered list.

Each step must be one of these forms:
  N. TOOL <tool_name>: <one-line reason>
  N. REASON: <one-line thought>
  N. RESPOND: <one-line description of the final answer>

Rules:
- Output ONLY the numbered list. No prose, no preamble, no code fences.
- Maximum {max_steps} steps. Plan must end with a RESPOND step.
- TOOL steps may only reference tools listed under "Available tools" below.
- Keep each step to one line — concise, actionable.

Example for "What time is it in Tokyo and what's 2+2?":
  1. TOOL get_current_time: get current time in Asia/Tokyo
  2. TOOL calculator: compute 2+2
  3. RESPOND: combine the two results in a single answer"""


def _generate_plan(prompt: str, chunks: list[dict], supports_tools: bool) -> tuple[str, int, int]:
    """Call Llama 8B to produce a numbered plan.

    Returns (plan_text, step_count, latency_ms). Empty plan_text on
    failure (caller treats as fail_open).
    """
    started = time.monotonic()

    # Available tools advertised to the planner. If the route doesn't
    # support tools, the planner should produce REASON/RESPOND only.
    if supports_tools:
        tool_list = ", ".join(t.name for t in TOOLS)
        tools_line = f"Available tools: {tool_list}\n"
    else:
        tools_line = "Available tools: (none — REASON and RESPOND only)\n"

    # Compact chunk preview — first line of each, capped at 5 chunks.
    if chunks:
        preview = "\n".join(
            f"  [{i+1}] {(c.get('text') or '').splitlines()[0][:120]}"
            for i, c in enumerate(chunks[:5])
        )
        chunks_block = f"Retrieved context preview:\n{preview}\n"
    else:
        chunks_block = "Retrieved context: (none)\n"

    user_msg = (
        f"User request: {prompt}\n\n"
        f"{tools_line}"
        f"{chunks_block}\n"
        f"Plan:"
    )

    cfg_trivial = ROUTE_REGISTRY["trivial"]
    client = ChatOpenAI(
        model=cfg_trivial["model_name"],
        base_url=cfg_trivial["url"],
        api_key="not-required",
        temperature=0.0,
        max_tokens=PLANNER_MAX_TOKENS,
        timeout=PLANNER_TIMEOUT_SECONDS,
    )
    try:
        response = client.invoke([
            SystemMessage(
                content=_PLANNER_SYSTEM_PROMPT.format(max_steps=PLANNER_MAX_STEPS)
            ),
            HumanMessage(content=user_msg),
        ])
        text = (response.content or "").strip()
    except Exception as e:
        log.warning("planner: LLM call failed: %s, fail-open", e)
        return ("", 0, int((time.monotonic() - started) * 1000))

    elapsed_ms = int((time.monotonic() - started) * 1000)

    # Count numbered lines as a rough step count for telemetry.
    # Strict parsing isn't needed — node_execute uses the raw text
    # as a system message hint, not as a structured plan.
    step_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            step_count += 1

    return (text, step_count, elapsed_ms)


def node_plan(state: AgentState) -> AgentState:
    """Run the planner LLM if enabled for this tier; fail-OPEN."""
    if not PLANNER_ENABLED:
        return {
            "plan_text": "",
            "plan_steps_count": 0,
            "planner_action": "skipped",
            "plan_ms": 0,
        }

    cfg = ROUTE_REGISTRY[state["route"]]
    if not cfg.get("use_planner"):
        return {
            "plan_text": "",
            "plan_steps_count": 0,
            "planner_action": "skipped",
            "plan_ms": 0,
        }

    text, count, ms = _generate_plan(
        state.get("prompt", ""),
        state.get("retrieved_chunks") or [],
        bool(cfg.get("supports_tools")),
    )

    if not text:
        # Fail-open: no plan produced, but the request still proceeds.
        # node_execute simply runs without the plan-system-message
        # injection.
        return {
            "plan_text": "",
            "plan_steps_count": 0,
            "planner_action": "fail_open",
            "plan_ms": ms,
        }

    log.info(
        "node_plan route=%s steps=%d ms=%d",
        state.get("route"), count, ms,
    )
    return {
        "plan_text": text,
        "plan_steps_count": count,
        "planner_action": "planned",
        "plan_ms": ms,
    }


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
    # Phase #15: pick stable vs canary variant for this request. Most
    # requests get the tier's default model_name; CANARY_<TIER>_FRACTION
    # of them go to the canary model. Recorded in state for telemetry +
    # response surfacing.
    variant_name, variant_label = _select_variant(state["route"])
    LG_VARIANT_TOTAL.labels(route=state["route"], variant=variant_label).inc()
    final_prompt = _build_rag_prompt(state["prompt"], state.get("retrieved_chunks", []))
    started = time.monotonic()

    if not cfg.get("supports_tools"):
        # Path B — legacy single-shot. Same code as before tool calling
        # was added; preserves behavior for 70B/DeepSeek tiers.
        # streaming=True (Phase #8): /invoke/stream's astream_events
        # picks up token-level events from this LLM call. /invoke (sync)
        # invoke() still works — LangChain collects the streamed chunks
        # and returns the full response as before. No behavior change
        # for non-streaming callers.
        client = ChatOpenAI(
            model=variant_name,  # Phase #15: stable or canary
            base_url=cfg["url"],
            api_key="not-required",
            max_tokens=state.get("max_tokens", 512),
            timeout=EXECUTE_TIMEOUT_SECONDS,
            streaming=True,
        )
        # Phase #13: prepend the plan as a system message on the
        # legacy path too. Same guidance pattern as path A. The plan
        # was generated assuming no tools; node_plan reflects that
        # by emitting REASON/RESPOND-only steps when supports_tools
        # is False.
        legacy_messages: list[BaseMessage] = []
        plan_text_b = state.get("plan_text") or ""
        if plan_text_b:
            legacy_messages.append(SystemMessage(
                content=(
                    "Suggested plan for this request (you may deviate "
                    "if you discover better steps):\n" + plan_text_b
                )
            ))
        legacy_messages.append(HumanMessage(content=final_prompt))
        response = client.invoke(legacy_messages)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return {
            "response": response.content or "",
            "execute_latency_ms": elapsed_ms,
            "tool_iterations": 0,
            "tool_calls_log": [],
            "tool_rate_limited_log": [],
            "variant_name": variant_name,
            "variant_label": variant_label,
        }

    # Path A — agentic loop with tool calling.
    # Bind tools so vLLM gets a tools=[...] payload on every request;
    # Llama 3.1 + llama3_json parser converts the model's tool-call
    # output into structured response.tool_calls. The bound client is
    # the SAME ChatOpenAI but with .bind_tools() applied — LangChain
    # plumbs the schema into the request.
    # streaming=True (Phase #8) enables token-level events for
    # /invoke/stream. Tool-calling + streaming compose: the model
    # streams tokens, and on a tool-call turn the streamed deltas
    # build up into a structured tool_calls field that LangChain
    # surfaces only when the turn finishes.
    client = ChatOpenAI(
        model=variant_name,  # Phase #15: stable or canary
        base_url=cfg["url"],
        api_key="not-required",
        max_tokens=state.get("max_tokens", 512),
        timeout=EXECUTE_TIMEOUT_SECONDS,
        streaming=True,
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
    ]
    # Phase #13: if a plan was produced (planner_action="planned"),
    # inject it as an additional system-level hint. The agent loop
    # still decides each tool call autonomously — the plan is
    # guidance, not a script. Empirically improves multi-step
    # completion on complex prompts.
    plan_text = state.get("plan_text") or ""
    if plan_text:
        messages.append(SystemMessage(
            content=(
                "Suggested plan for this request (you may deviate if "
                "you discover better steps):\n" + plan_text
            )
        ))
    messages.append(HumanMessage(content=final_prompt))

    # Set contextvars so search_session_docs sees the request's
    # session_id + auth_token. Reset in finally so we don't leak
    # one request's token into another request's context.
    sid_token = _AGENT_SESSION_ID.set(state.get("session_id"))
    auth_token_token = _AGENT_AUTH_TOKEN.set(state.get("auth_token"))

    tool_calls_log: list[str] = []
    tool_rate_limited_log: list[str] = []  # Phase #19
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
                    # Phase #19: per-(tool, user) rate-limit check.
                    # Denied tools return a synthetic ToolMessage so
                    # the agent loop sees it like any other tool error
                    # — model can decide to retry, fall back, or
                    # apologize. Better than hard-failing the request.
                    rl_user = state.get("user", "unknown")
                    rl_allowed, rl_remaining = _check_tool_rate_limit(name, rl_user)
                    if not rl_allowed:
                        cfg = TOOL_RATE_CONFIG.get(name, {})
                        window = cfg.get("window_seconds", 60)
                        result = (
                            f"tool {name} rate-limited (per-user); "
                            f"try again in <{window}s"
                        )
                        tool_rate_limited_log.append(name)
                        LG_TOOL_RATE_LIMITED_TOTAL.labels(tool=name).inc()
                        log.info(
                            "tool_rate_limited tool=%s user=%s window=%ds",
                            name, rl_user, window,
                        )
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
                model=variant_name,  # Phase #15: same variant as the loop
                base_url=cfg["url"],
                api_key="not-required",
                max_tokens=state.get("max_tokens", 512),
                timeout=EXECUTE_TIMEOUT_SECONDS,
                streaming=True,
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
        "tool_rate_limited_log": tool_rate_limited_log,
        "variant_name": variant_name,
        "variant_label": variant_label,
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
    g.add_node("input_validation", node_input_validation)
    g.add_node("safety_input", node_safety_input)
    g.add_node("pii_redact_input", node_pii_redact_input)
    g.add_node("cache_lookup", node_cache_lookup)
    g.add_node("load_memory", node_load_memory)
    g.add_node("rewrite_query", node_rewrite_query)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("ensure_warm", node_ensure_warm)
    g.add_node("plan", node_plan)
    g.add_node("execute", node_execute)
    g.add_node("reflect", node_reflect)
    g.add_node("safety_output", node_safety_output)
    g.add_node("hallucination_check", node_hallucination_check)
    g.add_node("pii_redact_output", node_pii_redact_output)
    g.add_node("cache_store", node_cache_store)
    g.add_node("save_memory", node_save_memory)
    g.add_edge(START, "budget_check")
    g.add_conditional_edges(
        "budget_check",
        _route_after_budget_check,
        # Phase #20: route to input_validation first, which then
        # conditionally hands off to safety_input or short-circuits END.
        {"input_validation": "input_validation", END: END},
    )
    g.add_conditional_edges(
        "input_validation",
        _route_after_input_validation,
        {"safety_input": "safety_input", END: END},
    )
    g.add_conditional_edges(
        "safety_input",
        _route_after_safety_input,
        # Routes to pii_redact_input first; that produces the redacted
        # prompt used by cache_lookup as the embedding key, and by
        # downstream retrieve / rewrite / save_memory.
        {"classify": "pii_redact_input", END: END},
    )
    g.add_edge("pii_redact_input", "cache_lookup")
    g.add_conditional_edges(
        "cache_lookup",
        _route_after_cache_lookup,
        # HIT → save_memory (response is already in state, skip pipeline).
        # MISS → load_memory (full pipeline).
        {"load_memory": "load_memory", "save_memory": "save_memory"},
    )
    g.add_edge("load_memory", "rewrite_query")
    g.add_edge("rewrite_query", "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "ensure_warm")
    g.add_edge("ensure_warm", "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "reflect")
    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"retrieve": "retrieve", END: "safety_output"},
    )
    # safety_output → hallucination_check → pii_redact_output →
    # cache_store → save_memory → END.
    #
    # Order rationale: hallucination_check needs the unredacted
    # response to compare claims against retrieved chunks (redaction
    # would lose info). pii_redact_output runs AFTER, then cache_store
    # and save_memory persist the redacted version — preventing PII
    # from being cached or memorized.
    g.add_edge("safety_output", "hallucination_check")
    g.add_edge("hallucination_check", "pii_redact_output")
    g.add_edge("pii_redact_output", "cache_store")
    g.add_edge("cache_store", "save_memory")
    g.add_edge("save_memory", END)
    return g.compile()


GRAPH = build_graph()


# --- Models ----------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """User feedback on a specific /invoke response.

    The trace_id is the langfuse_trace_id returned in the InvokeResponse
    (also surfaced in /invoke/stream's done event). Bind feedback to the
    specific response, not just the session, so analytics can cleanly
    correlate cause (specific prompt + retrieved chunks + tool calls +
    safety/cache verdicts) with effect (user rating).
    """
    trace_id: str = Field(..., description="langfuse_trace_id from the InvokeResponse")
    # Two rating styles supported. Most chat UIs use thumbs (binary);
    # advanced UIs surface 1-5 scales for granularity. Accept both,
    # normalize on read for stats.
    rating: Literal["up", "down"] = Field(..., description="Thumbs-up/down rating")
    # Optional 1-5 scale, used in addition to rating for nuanced surveys.
    score: Optional[int] = Field(default=None, ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=2000)
    # Free-form tags useful in product (e.g. ["accuracy", "tone",
    # "missing_citation", "hallucinated"]). Indexed in the future for
    # category-specific aggregates.
    categories: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    ok: bool
    trace_id: str
    user: str
    persisted_at: float
    langfuse_recorded: bool


class FeedbackStatsResponse(BaseModel):
    """Aggregate feedback stats — operator-facing.

    Per-user OR per-session (caller chooses via query param). Returns
    the most recent N feedback entries plus thumbs-up/down counts.
    Bigger analytics (time-series, category breakdowns) belong in
    Grafana on top of Langfuse traces; this endpoint is just for
    quick health checks.
    """
    user: str
    total: int
    up: int
    down: int
    recent: list[dict]


class SessionExportResponse(BaseModel):
    """All per-(user, session_id) data this service holds.

    Shape:
      user, session_id        identity
      memory_turns            list of turn dicts (prompt, response, ts)
      memory_summary          long-term conversation summary string
      cache_entries           list of cache entries with embeddings
                              stripped (1024-dim float arrays are
                              useless to users + minor signal leak)

    Excluded from export (separate concerns):
      Qdrant per-session chunks         in rag-service / Qdrant —
                                         use rag-service's own export
                                         path or qdrant API directly.
      Langfuse traces                    in Langfuse — export via
                                         Langfuse SDK or UI.
      Feedback records                   user-scoped, not session-
                                         scoped today; covered by a
                                         future /user/me/export.
    """
    user: str
    session_id: str
    memory_turns: list[dict]
    memory_summary: str
    cache_entries: list[dict]
    # Counts surfaced separately so callers can verify "we got
    # everything" without iterating list lengths.
    memory_turn_count: int
    cache_entry_count: int


class SessionDeleteResponse(BaseModel):
    """Right-to-deletion confirmation.

    deleted_keys is the total number of Redis keys actually deleted
    (across memory turns key, memory summary key, cache index, and
    each cache entry hash). Operator can sanity-check by calling
    /session/<id>/export immediately after — should return empty
    structures.
    """
    user: str
    session_id: str
    deleted_keys: int


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
    # Phase #7: semantic prompt cache telemetry. cache_hit drives the
    # chat-ui "served from cache" badge; cache_similarity helps tune
    # CACHE_SIMILARITY_THRESHOLD (watch the distribution to see how
    # close hits cluster around the boundary).
    cache_hit: bool = False
    cache_similarity: float = 0.0
    cache_lookup_ms: int = 0
    cache_store_ms: int = 0
    # Phase #9: hallucination-detection telemetry. action is the
    # terminal disposition for the caller; verdict + confidence are
    # the raw grader output for observability.
    hallucination_action: str = "disabled"
    hallucination_verdict: str = "skipped"
    hallucination_confidence: float = 0.0
    hallucination_check_ms: int = 0
    # Phase #11: PII redaction telemetry. entities_found is a dict of
    # entity_type → count (e.g. {"email": 1, "ipv4": 2}). Original
    # values are NEVER returned — only the counts and types — to
    # avoid defeating the redaction by leaking via the API response.
    pii_redact_action: str = "skipped"
    pii_entities_found: dict = Field(default_factory=dict)
    pii_redact_ms: int = 0
    # Phase #16: input-side PII redaction telemetry. Same shape as
    # the output-side fields, prefix _input distinguishes which side.
    pii_input_action: str = "skipped"
    pii_input_entities_found: dict = Field(default_factory=dict)
    pii_input_redact_ms: int = 0
    # Phase #13: planner output. plan_text is the raw numbered list
    # the planner produced (empty when planner skipped). chat-ui can
    # render it as a "Show plan" expandable section so users see
    # the model's intended approach before tools fire.
    planner_action: str = "skipped"
    plan_text: str = ""
    plan_steps_count: int = 0
    plan_ms: int = 0
    # Phase #15: A/B canary variant. variant_name is the actual model
    # name node_execute used (may differ from the tier's default).
    # variant_label is "stable" | "canary" — surfaced for downstream
    # eval correlation. The /feedback endpoint can record variant
    # alongside ratings to enable per-variant satisfaction analysis.
    variant_name: str = ""
    variant_label: str = "stable"
    # Phase #19: per-tool rate-limit telemetry. Names of tools the agent
    # tried to invoke but got blocked by per-(tool, user) rate limits.
    # chat-ui can render "your http_fetch usage hit the per-minute cap"
    # alongside the response.
    tool_rate_limited_log: list[str] = Field(default_factory=list)
    # Phase #20: input validation telemetry. action drives chat-ui
    # rendering ("we couldn't process your prompt — try shorter");
    # details surfaces the structured reason for debugging without
    # leaking the offending content.
    input_validation_action: str = "skipped"
    input_validation_details: dict = Field(default_factory=dict)


# --- Routes ----------------------------------------------------------------

def _build_invoke_response_dict(
    final_state: dict,
    user: str,
    trace_id: Optional[str],
) -> dict:
    """Shape final graph state into the InvokeResponse field set, as a dict.

    Shared by /invoke (sync, returns InvokeResponse) and /invoke/stream
    (SSE done event, JSON-serialized via json.dumps). Keeping ONE source
    of truth for the response shape avoids drift when new fields land.
    """
    return {
        "response": final_state.get("response", ""),
        "route": final_state.get("route", "trivial"),
        "cold_start": final_state.get("cold_start", False),
        "warm_wait_seconds": final_state.get("warm_wait_seconds", 0.0),
        "execute_latency_ms": final_state.get("execute_latency_ms", 0),
        "classifier_raw": final_state.get("classifier_raw", ""),
        "user": user,
        "retrieve_count": final_state.get("retrieve_count", 0),
        "retrieve_ms": final_state.get("retrieve_ms", 0),
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "langfuse_trace_id": trace_id,
        "tool_iterations": final_state.get("tool_iterations", 0),
        "tool_calls_log": final_state.get("tool_calls_log", []),
        "reasoning_cycles": final_state.get("cycles", 0),
        "reflection_log": final_state.get("reflection_log", []),
        "safety_action": final_state.get("safety_action", "passed"),
        "safety_input_verdict": final_state.get("safety_input_verdict", "skipped"),
        "safety_output_verdict": final_state.get("safety_output_verdict", "skipped"),
        "safety_categories": final_state.get("safety_categories", []),
        "safety_input_ms": final_state.get("safety_input_ms", 0),
        "safety_output_ms": final_state.get("safety_output_ms", 0),
        "budget_action": final_state.get("budget_action", "disabled"),
        "budget_consumed": final_state.get("budget_consumed", 0),
        "budget_remaining": final_state.get("budget_remaining", 0),
        "query_rewritten": (
            final_state.get("refined_query", "") or ""
            if final_state.get("query_rewritten") else ""
        ),
        "query_rewrite_ms": final_state.get("query_rewrite_ms", 0),
        "memory_turn_count": len(final_state.get("memory_recent_turns") or []),
        "memory_load_ms": final_state.get("memory_load_ms", 0),
        "memory_save_ms": final_state.get("memory_save_ms", 0),
        "cache_hit": final_state.get("cache_hit", False),
        "cache_similarity": final_state.get("cache_similarity", 0.0),
        "cache_lookup_ms": final_state.get("cache_lookup_ms", 0),
        "cache_store_ms": final_state.get("cache_store_ms", 0),
        "hallucination_action": final_state.get("hallucination_action", "disabled"),
        "hallucination_verdict": final_state.get("hallucination_verdict", "skipped"),
        "hallucination_confidence": final_state.get("hallucination_confidence", 0.0),
        "hallucination_check_ms": final_state.get("hallucination_check_ms", 0),
        "pii_redact_action": final_state.get("pii_redact_action", "skipped"),
        "pii_entities_found": final_state.get("pii_entities_found") or {},
        "pii_redact_ms": final_state.get("pii_redact_ms", 0),
        "pii_input_action": final_state.get("pii_input_action", "skipped"),
        "pii_input_entities_found": final_state.get("pii_input_entities_found") or {},
        "pii_input_redact_ms": final_state.get("pii_input_redact_ms", 0),
        "planner_action": final_state.get("planner_action", "skipped"),
        "plan_text": final_state.get("plan_text", ""),
        "plan_steps_count": final_state.get("plan_steps_count", 0),
        "plan_ms": final_state.get("plan_ms", 0),
        "variant_name": final_state.get("variant_name", ""),
        "variant_label": final_state.get("variant_label", "stable"),
        "tool_rate_limited_log": final_state.get("tool_rate_limited_log") or [],
        "input_validation_action": final_state.get("input_validation_action", "skipped"),
        "input_validation_details": final_state.get("input_validation_details") or {},
    }


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Shallow liveness probe — process is up + answering. Used by
    livenessProbe; doesn't check dependencies (a Redis outage shouldn't
    restart the langgraph-service pod, just degrade its serving)."""
    return {"ok": True, "service": "langgraph-service", "version": app.version}


@app.get("/readyz", include_in_schema=False)
def readyz() -> dict:
    """Deep readiness probe — confirms required deps are reachable.

    Required (failure → 503, pulls pod out of Service rotation):
      - langgraph-redis (budget, cache, memory, feedback all need it)
      - Keycloak JWKs endpoint (every /invoke does JWT validation)

    Soft (logged in the response, but not gated):
      - vllm-llama-8b — JIT-scaled, may legitimately be at 0 replicas;
        a single /invoke that routes to trivial would scale it up.
        Soft check so cold-tier state doesn't pull pods out of rotation.
      - vllm-llama-guard-3-8b — same scale-to-zero pattern when safety
        filter isn't active. Soft.
      - vllm-bge-m3 — same pattern when cache isn't active. Soft.
      - ingestion-service — same pattern; not critical for /invoke
        flows that don't /upload.

    Returns:
      200 with {"ready": true, "checks": {...}} when all required deps
            pass. Soft check failures are visible in the body.
      503 with {"ready": false, "checks": {...}} when any required dep
            fails. K8s pulls the pod out of the Service.
    """
    checks: dict = {}
    overall_ok = True

    # --- Required: Redis (budget/cache/memory/feedback all depend) ---
    redis_client = _get_redis()
    if redis_client is None:
        checks["redis"] = {"required": True, "ok": False, "detail": "client not initialized"}
        overall_ok = False
    else:
        try:
            redis_client.ping()
            checks["redis"] = {"required": True, "ok": True}
        except Exception as e:
            checks["redis"] = {
                "required": True, "ok": False, "detail": str(e)[:120]
            }
            overall_ok = False

    # --- Required: Keycloak JWKs (every /invoke validates JWT) ---
    try:
        keys = _fetch_jwks()
        checks["keycloak_jwks"] = {
            "required": True,
            "ok": bool(keys),
            "key_count": len(keys),
        }
        if not keys:
            overall_ok = False
    except Exception as e:
        checks["keycloak_jwks"] = {
            "required": True, "ok": False, "detail": str(e)[:120]
        }
        overall_ok = False

    # --- Soft: in-cluster model endpoints. Each is allowed to be cold. ---
    soft_endpoints = {
        "vllm_llama_8b": f"{ROUTE_REGISTRY['trivial']['url'].rsplit('/v1', 1)[0]}/health",
        "vllm_llama_guard": f"{SAFETY_LLAMA_GUARD_URL.rsplit('/v1', 1)[0]}/health",
        "vllm_bge_m3": f"{CACHE_EMBEDDINGS_URL.rsplit('/v1', 1)[0]}/health",
        "rag_service": f"{RAG_SERVICE_URL}/healthz",
    }
    for name, url in soft_endpoints.items():
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(url)
                checks[name] = {
                    "required": False,
                    "ok": resp.status_code < 500,
                    "status": resp.status_code,
                }
        except Exception as e:
            # Soft — failure is expected when the model is at
            # replicas=0. Record but don't gate on it.
            checks[name] = {
                "required": False,
                "ok": False,
                "detail": str(e)[:80],
            }

    payload = {
        "ready": overall_ok,
        "service": "langgraph-service",
        "version": app.version,
        "checks": checks,
    }
    if not overall_ok:
        # Return body in detail so kubectl describe shows what's wrong
        # in the event log when readiness fails. JSON-encode here
        # because HTTPException's detail can be a dict.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=payload,
        )
    return payload


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

    # Phase #14: Prometheus metric emission. Best-effort; failures
    # logged but don't fail the request. Mirrored in /invoke/stream's
    # done branch so both endpoints contribute the same Counters +
    # Histograms.
    _emit_request_metrics(dict(final_state))

    # Shape via the shared helper (also used by /invoke/stream's done
    # event), then wrap retrieved_chunks dicts in the typed model. The
    # rest expand straight into InvokeResponse via **.
    payload = _build_invoke_response_dict(final_state, user, trace_id)
    payload["retrieved_chunks"] = [
        RetrievedChunkOut(**c) for c in payload["retrieved_chunks"]
    ]
    return InvokeResponse(**payload)


# --- /invoke/stream — SSE endpoint (Phase #8) -----------------------------
#
# Same input shape as /invoke. Different output: text/event-stream with
# the following SSE event types:
#
#   event: node_start
#   data: {"node": "<name>"}
#       Emitted when a graph node begins executing. Useful for
#       chat-ui progress indicators ("Loading memory...", "Routing...",
#       "Retrieving documents...").
#
#   event: node_end
#   data: {"node": "<name>", "<selected fields>": ...}
#       Emitted when a graph node finishes. Carries a small selected
#       subset of state fields per node (e.g. classify → route,
#       cache_lookup → cache_hit/cache_similarity, retrieve →
#       retrieve_count). Cherry-picked rather than dumping the whole
#       state because state can be large (retrieved_chunks alone may
#       be ~50 KB).
#
#   event: token
#   data: {"content": "<delta>"}
#       Emitted for each token chunk produced by an LLM streaming
#       call inside node_execute. Filtered to ONLY the user-facing
#       response — internal LLM calls (rewrite_query, classify,
#       summarize_memory, llama_guard_*, reflect) don't emit token
#       events because their outputs aren't user-facing.
#
#   event: done
#   data: <full InvokeResponse JSON>
#       Final event, sent once the graph completes. Same shape as the
#       sync /invoke response. Lets the UI render final telemetry
#       (latency breakdown, tool calls log, citation chunks, etc.)
#       without re-fetching.
#
#   event: error
#   data: {"detail": "<msg>", "status": <code>}
#       Sent on graph exception or auth failure (sets the SSE stream's
#       last event to error rather than done; client must check).
#
# Implementation note: streaming=True on the ChatOpenAI clients in
# node_execute (Phase #8a) is what makes on_chat_model_stream events
# fire. The OTHER ChatOpenAI clients (rewrite_query, classify, etc.)
# do NOT have streaming=True, so they don't emit token events — by
# design.


_NODE_NAMES = {
    "budget_check", "input_validation", "safety_input",
    "pii_redact_input", "cache_lookup", "load_memory",
    "rewrite_query", "classify", "retrieve", "ensure_warm",
    "plan", "execute", "reflect", "safety_output",
    "hallucination_check", "pii_redact_output",
    "cache_store", "save_memory",
}

# Per-node selection of which state fields are interesting enough to
# include in the node_end event. Matches the InvokeResponse shape but
# scoped per-node so the SSE stream stays small.
_NODE_END_FIELDS = {
    "budget_check": ["budget_action", "budget_consumed", "budget_remaining"],
    "input_validation": ["input_validation_action", "input_validation_details"],
    "safety_input": ["safety_input_verdict", "safety_categories", "safety_action"],
    "cache_lookup": ["cache_hit", "cache_similarity"],
    "load_memory": ["memory_turn_count"],
    "rewrite_query": ["query_rewritten"],
    "classify": ["route", "classifier_raw"],
    "retrieve": ["retrieve_count", "retrieve_ms"],
    "ensure_warm": ["cold_start", "warm_wait_seconds"],
    "plan": ["planner_action", "plan_steps_count"],
    "execute": [
        "execute_latency_ms", "tool_iterations", "tool_calls_log",
        "variant_name", "variant_label",
    ],
    "reflect": ["cycles", "needs_more_context"],
    "safety_output": ["safety_output_verdict", "safety_action"],
    "hallucination_check": [
        "hallucination_action", "hallucination_verdict", "hallucination_confidence",
    ],
    "pii_redact_input": ["pii_input_action", "pii_input_entities_found"],
    "pii_redact_output": ["pii_redact_action", "pii_entities_found"],
    "cache_store": [],
    "save_memory": [],
}


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Events frame.

    SSE wire format: lines, blank-line terminator. `event:` field
    sets the event type the browser dispatches; `data:` carries the
    payload (here, JSON-encoded). Multi-line data is allowed by
    repeating `data:` lines, but our payloads are single-line JSON.
    """
    import json as _json
    return f"event: {event}\ndata: {_json.dumps(data)}\n\n"


@app.post("/invoke/stream")
async def invoke_stream(
    req: InvokeRequest,
    claims: Annotated[dict, Depends(require_jwt)],
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)],
) -> StreamingResponse:
    """SSE variant of /invoke.

    Same auth, same input shape, same graph. Streams progress events
    as the graph executes plus token-level deltas as the model
    generates. Sends a final `done` event with the full InvokeResponse
    shape so the UI can render telemetry on completion.
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    log.info(
        "invoke_stream",
        extra={
            "user": user,
            "prompt_len": len(req.prompt),
            "session_id": req.session_id or "",
        },
    )
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
        import uuid as _uuid
        trace_id = _uuid.uuid4().hex
        config = {
            "callbacks": [_LANGFUSE_CB],
            "metadata": {
                "langfuse_user_id": user,
                "langfuse_tags": ["langgraph-service", "stream"],
                "langfuse_trace_id": trace_id,
            },
        }

    async def event_stream():
        # Accumulate per-node return dicts so we can build the final
        # InvokeResponse-shaped done event. astream_events fires an
        # on_chain_end per node with the partial state that node
        # returned; we merge them in arrival order, which matches
        # graph topology.
        accumulated: dict = dict(initial)
        try:
            async for event in GRAPH.astream_events(
                initial, config=config, version="v2"
            ):
                ev_type = event.get("event")
                name = event.get("name", "")

                if ev_type == "on_chain_start" and name in _NODE_NAMES:
                    yield _sse("node_start", {"node": name})

                elif ev_type == "on_chain_end" and name in _NODE_NAMES:
                    out = event.get("data", {}).get("output") or {}
                    if isinstance(out, dict):
                        accumulated.update(out)
                    payload = {"node": name}
                    for field in _NODE_END_FIELDS.get(name, []):
                        if field in out:
                            payload[field] = out[field]
                    yield _sse("node_end", payload)

                elif ev_type == "on_chat_model_stream":
                    # Filter token events to ONLY those from node_execute.
                    # The metadata's langgraph_node tells us which graph
                    # node this LLM call is running inside.
                    md = event.get("metadata", {}) or {}
                    if md.get("langgraph_node") != "execute":
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    content = getattr(chunk, "content", None) or ""
                    if content:
                        yield _sse("token", {"content": content})

            # Phase #14: emit metrics from the accumulated terminal
            # state — same call as the sync /invoke handler does.
            _emit_request_metrics(accumulated)

            # Final event: full response shape so the UI gets all
            # the telemetry without a follow-up call.
            payload = _build_invoke_response_dict(accumulated, user, trace_id)
            yield _sse("done", payload)
        except RuntimeError as e:
            # ensure_warm failure path — same as /invoke's 502 mapping
            log.error("invoke_stream graph runtime error: %s", e)
            yield _sse("error", {"status": 502, "detail": str(e)[:300]})
        except Exception as e:
            log.exception("invoke_stream unexpected error")
            yield _sse("error", {"status": 500, "detail": str(e)[:300]})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent intermediate proxies (NGINX, Istio) from buffering
            # — buffering kills the streaming UX. X-Accel-Buffering=no
            # is NGINX-specific; no harm on other proxies. Cache-Control
            # disables CDN caching of an SSE stream.
            "Cache-Control": "no-cache, no-store, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# --- /feedback endpoints (Phase #10) ---------------------------------------


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(
    req: FeedbackRequest,
    claims: Annotated[dict, Depends(require_jwt)],
) -> FeedbackResponse:
    """Submit feedback for a specific /invoke response.

    Dual-write: Redis (operational store, queryable via /feedback/stats)
    + Langfuse score (visible in trace UI alongside the original
    request). Redis is authoritative; Langfuse is the dashboard view.

    Auth binds (user, trace_id): only the user who made the original
    request can submit feedback for it. Trace_id ownership isn't
    server-validated against the original /invoke caller (would
    require persisting trace→user mapping); instead, the per-user
    Redis key prefix means a user's feedback can only land under
    their own namespace, regardless of which trace_id they submit.
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"

    redis_client = _get_redis()
    if redis_client is None:
        # Feedback loss is a quality-bar regression — surface as 503
        # rather than swallow.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="feedback store unavailable",
        )

    import json as _json
    ts = time.time()
    record = {
        "trace_id": req.trace_id,
        "rating": req.rating,
        "score": req.score if req.score is not None else "",
        "comment": (req.comment or "")[:2000],
        "categories": _json.dumps(req.categories or []),
        "user": user,
        "ts": str(ts),
    }
    feedback_key = _feedback_key(user, req.trace_id)
    index_key = _feedback_index_key(user)

    try:
        # Pipelined: hash write + index zadd + LRU trim + TTLs
        pipe = redis_client.pipeline(transaction=False)
        pipe.hset(feedback_key, mapping=record)
        pipe.expire(feedback_key, FEEDBACK_TTL_SECONDS)
        pipe.zadd(index_key, {req.trace_id: ts})
        # Keep only the most recent FEEDBACK_INDEX_MAX_ENTRIES in the
        # per-user index. Older entries are still in their hash keys
        # (each with its own TTL), but the index list stays bounded.
        pipe.zremrangebyrank(index_key, 0, -(FEEDBACK_INDEX_MAX_ENTRIES + 1))
        pipe.expire(index_key, FEEDBACK_TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        log.error("feedback: redis op failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="feedback persist failed",
        )

    # Best-effort Langfuse score. Failure here doesn't fail the request
    # — Redis is authoritative, Langfuse is observability sugar.
    langfuse_recorded = False
    lf_client = _get_langfuse_client()
    if lf_client is not None:
        try:
            # Numeric score: 1.0 for thumbs-up, 0.0 for thumbs-down.
            # Plus the 1-5 score on a separate name if provided.
            lf_client.create_score(
                trace_id=req.trace_id,
                name="user_feedback",
                value=1.0 if req.rating == "up" else 0.0,
                comment=req.comment or None,
            )
            if req.score is not None:
                lf_client.create_score(
                    trace_id=req.trace_id,
                    name="user_feedback_1to5",
                    value=float(req.score),
                )
            langfuse_recorded = True
        except Exception as e:
            log.warning("feedback: langfuse score emit failed: %s", e)

    log.info(
        "feedback user=%s trace=%s rating=%s score=%s langfuse=%s",
        user, req.trace_id, req.rating, req.score, langfuse_recorded,
    )

    return FeedbackResponse(
        ok=True,
        trace_id=req.trace_id,
        user=user,
        persisted_at=ts,
        langfuse_recorded=langfuse_recorded,
    )


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats(
    claims: Annotated[dict, Depends(require_jwt)],
    limit: int = 25,
) -> FeedbackStatsResponse:
    """Operator-facing feedback summary for the calling user.

    Returns total count, thumbs-up/down split, and the most recent N
    feedback entries (default 25). Per-user only — no admin path
    surfaces other users' feedback (would need its own auth scope).

    For richer analytics (time-series, category breakdowns, regression
    correlation) use Langfuse's UI — every score submitted via
    /feedback is also there.
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    limit = max(1, min(limit, FEEDBACK_INDEX_MAX_ENTRIES))

    redis_client = _get_redis()
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="feedback store unavailable",
        )

    index_key = _feedback_index_key(user)
    import json as _json

    try:
        # ZREVRANGE for newest-first ordering
        trace_ids: list[str] = redis_client.zrevrange(index_key, 0, limit - 1) or []
        if not trace_ids:
            return FeedbackStatsResponse(
                user=user, total=0, up=0, down=0, recent=[]
            )
        # Pipeline: hgetall for each entry + zcard for total
        pipe = redis_client.pipeline(transaction=False)
        for tid in trace_ids:
            pipe.hgetall(_feedback_key(user, tid))
        pipe.zcard(index_key)
        results = pipe.execute()
    except Exception as e:
        log.error("feedback_stats: redis op failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="feedback query failed",
        )

    entries = results[:-1]
    total = int(results[-1])
    recent: list[dict] = []
    up = 0
    down = 0
    for e in entries:
        if not e:
            continue
        rec = {
            "trace_id": e.get("trace_id", ""),
            "rating": e.get("rating", ""),
            "score": e.get("score") or None,
            "comment": e.get("comment") or "",
            "ts": float(e.get("ts", "0") or 0),
        }
        try:
            rec["categories"] = _json.loads(e.get("categories") or "[]")
        except (_json.JSONDecodeError, TypeError):
            rec["categories"] = []
        if rec["rating"] == "up":
            up += 1
        elif rec["rating"] == "down":
            down += 1
        recent.append(rec)

    # NOTE: up/down counts here only reflect the most-recent `limit`
    # entries, not the full history. For accurate global counts the
    # operator should aggregate via Langfuse score export. Document
    # in the response shape if/when this becomes confusing.

    return FeedbackStatsResponse(
        user=user, total=total, up=up, down=down, recent=recent
    )


# --- /session export + delete (Phase #17) ----------------------------------
#
# GDPR/CCPA-grade right-to-deletion + right-to-export for per-(user,
# session_id) data this service holds. Companion to Phase #16's PII
# redaction — together they cover the data-handling story:
#
#   #16 prevents PII from being persisted in the first place
#       (storage layers see redacted prompts only).
#   #17 lets users see what IS persisted, and request its deletion.
#
# Auth: same Keycloak JWT as /invoke. Per-user namespacing in Redis
# keys (`mem:<user>:*`, `cache:<user>:*`) means a malicious user can
# only export/delete data under their own namespace; a forged
# session_id submission lands in the calling user's keyspace.
#
# Excluded from these endpoints (separate concerns documented in the
# response models):
#   Qdrant per-session chunks         rag-service responsibility
#   Langfuse traces                   Langfuse SDK/UI export
#   User-level feedback records       future /user/me/* endpoints


@app.get("/session/{session_id}/export", response_model=SessionExportResponse)
def session_export(
    session_id: str,
    claims: Annotated[dict, Depends(require_jwt)],
) -> SessionExportResponse:
    """Return all per-(user, session) data held by this service.

    Memory turns + summary, cache entries (with embeddings stripped).
    Per-user namespacing means the calling user can only see their
    own data; submitting another user's session_id returns empty
    structures (their data is under a different Redis keyspace).
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    redis_client = _get_redis()
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="session store unavailable",
        )

    import json as _json

    try:
        # Memory: turns list (LRANGE) + summary (GET) in one round-trip
        pipe = redis_client.pipeline(transaction=False)
        pipe.lrange(_memory_turns_key(user, session_id), 0, -1)
        pipe.get(_memory_summary_key(user, session_id))
        pipe.zrange(_cache_index_key(user, session_id), 0, -1)
        results = pipe.execute()
        raw_turns: list[str] = results[0] or []
        summary: str = results[1] or ""
        cache_entry_ids: list[str] = results[2] or []
    except Exception as e:
        log.error("session_export: redis op failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="session export failed",
        )

    memory_turns: list[dict] = []
    for raw in raw_turns:
        try:
            memory_turns.append(_json.loads(raw))
        except (_json.JSONDecodeError, TypeError):
            continue

    # Fetch cache entry hashes in one pipeline. Drop the embedding
    # field from each — 1024 floats is useless data to a user and a
    # minor signal leak (you could partially reverse-engineer what
    # bge-m3's embedding represents).
    cache_entries: list[dict] = []
    if cache_entry_ids:
        try:
            pipe = redis_client.pipeline(transaction=False)
            for eid in cache_entry_ids:
                pipe.hgetall(_cache_entry_key(user, session_id, eid))
            entries = pipe.execute()
        except Exception as e:
            log.warning("session_export: cache hgetall failed: %s", e)
            entries = []
        for entry in entries:
            if not entry:
                continue
            entry.pop("embedding", None)
            cache_entries.append(entry)

    log.info(
        "session_export user=%s session=%s turns=%d cache=%d",
        user, session_id, len(memory_turns), len(cache_entries),
    )

    return SessionExportResponse(
        user=user,
        session_id=session_id,
        memory_turns=memory_turns,
        memory_summary=summary,
        cache_entries=cache_entries,
        memory_turn_count=len(memory_turns),
        cache_entry_count=len(cache_entries),
    )


@app.delete("/session/{session_id}", response_model=SessionDeleteResponse)
def session_delete(
    session_id: str,
    claims: Annotated[dict, Depends(require_jwt)],
) -> SessionDeleteResponse:
    """Atomically wipe all per-(user, session) data.

    Deletes memory turns key, memory summary key, cache index key,
    and every cache entry hash whose ID is in the index. Single
    Redis pipeline with transaction=True so the delete either
    completes fully or not at all (no partial-state risk where
    memory's gone but cache lingers).

    Right-to-deletion sanity check: after a successful 200, calling
    /session/<id>/export should return empty structures
    (memory_turns=[], cache_entries=[], memory_summary=""). If the
    operator wants belt-and-braces verification, that's a one-line
    follow-up curl.

    NOT scoped within this endpoint:
      - Qdrant chunks for this session_id (separate service).
      - Langfuse traces (separate dashboard).
      - User-level feedback records (would require a separate
        /user/me/delete endpoint with admin-style scope).
    """
    user = claims.get("preferred_username") or claims.get("sub") or "unknown"
    redis_client = _get_redis()
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="session store unavailable",
        )

    cache_index_key = _cache_index_key(user, session_id)

    try:
        # First fetch cache entry IDs so we know which entry hashes
        # to delete. ZRANGE is non-mutating; safe to do outside the
        # transaction.
        cache_entry_ids: list[str] = redis_client.zrange(
            cache_index_key, 0, -1
        ) or []

        # Build the full delete list:
        #   memory:turns, memory:summary, cache:index, each cache:entry
        keys_to_delete = [
            _memory_turns_key(user, session_id),
            _memory_summary_key(user, session_id),
            cache_index_key,
        ]
        for eid in cache_entry_ids:
            keys_to_delete.append(_cache_entry_key(user, session_id, eid))

        # Atomic MULTI/EXEC delete — either all keys go or none do.
        # Redis DEL on a non-existent key returns 0, not an error,
        # so we can safely include keys that may or may not exist.
        pipe = redis_client.pipeline(transaction=True)
        if keys_to_delete:
            pipe.delete(*keys_to_delete)
        results = pipe.execute()
        # results[0] is the integer count of keys actually deleted.
        deleted_count = int(results[0]) if results else 0
    except Exception as e:
        log.error("session_delete: redis op failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="session delete failed",
        )

    log.info(
        "session_delete user=%s session=%s deleted_keys=%d",
        user, session_id, deleted_count,
    )

    return SessionDeleteResponse(
        user=user,
        session_id=session_id,
        deleted_keys=deleted_count,
    )
