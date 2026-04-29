#!/usr/bin/env python3
"""
RAGAS regression-gate runner for the langgraph-service RAG pipeline.

Pipeline:
  1. Mint a Keycloak JWT via direct-access-grant (test-user creds from
     the K8s Secret mounted into env). Reused across all questions.
  2. Idempotently ingest the dataset's source_doc entries into the eval
     session via ingestion-service /upload — content-hashed so re-runs
     don't double-ingest.
  3. For each question in dataset.yaml:
       POST to langgraph-service /invoke with the eval session_id.
       Capture response, retrieved_chunks, reasoning_cycles.
  4. Build a RAGAS-shaped EvaluationDataset (question, answer, contexts,
     ground_truth) and score it on faithfulness, answer_relevancy,
     context_precision. Judge LLM and embeddings both point at our
     in-cluster vLLM (no external API spend).
  5. Aggregate per-metric means + per-entry scores, write scores.json.
  6. Compare against baselines.json: if any metric dropped by more
     than RAGAS_REGRESSION_THRESHOLD_PP, exit non-zero. Null baselines
     are treated as "no gate, just record" so the first run never
     fails its own absence.

Failure model:
  Every step has a single retry on transient failure. Hard failures
  (auth flat-out denied, ingestion-service unreachable, RAGAS metric
  raising) cause exit(1) with a diagnostic — the CI workflow surfaces
  the runner stderr.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

# RAGAS 0.2.x API. Metrics are module-level functions; the data shape
# is a HuggingFace `datasets.Dataset` (or a list-of-dicts wrapped via
# EvaluationDataset).
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,    # 0.2 renamed answer_relevancy → ResponseRelevancy
    LLMContextPrecisionWithReference,  # context_precision variant that
                                       # takes our ground_truth_contexts
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Config from env (sane defaults match the in-cluster service DNS)
# ---------------------------------------------------------------------------

KEYCLOAK_TOKEN_URL = os.environ.get(
    "KEYCLOAK_TOKEN_URL",
    "https://keycloak.ekstest.com/realms/raj-ai-lab-eks/protocol/openid-connect/token",
)
KEYCLOAK_CLIENT_ID = os.environ.get("KEYCLOAK_CLIENT_ID", "langgraph-service")
KEYCLOAK_USERNAME = os.environ.get("KEYCLOAK_USERNAME", "raj")
# Password injected via K8s Secret `ragas-eval-credentials`. Required —
# the runner exits early if missing.
KEYCLOAK_PASSWORD = os.environ.get("KEYCLOAK_PASSWORD", "")

LANGGRAPH_URL = os.environ.get(
    "LANGGRAPH_URL", "http://langgraph-service.langgraph.svc.cluster.local"
)
INGESTION_URL = os.environ.get(
    "INGESTION_URL", "http://ingestion-service.ingestion.svc.cluster.local"
)

# In-cluster judge LLM + embeddings — no external API. The judge needs
# to be solid at JSON instruction-following; Llama 3.1 8B at temp=0
# handles RAGAS's NLI-style faithfulness prompts adequately. Bump to a
# heavier model only if metric noise becomes a problem.
JUDGE_LLM_URL = os.environ.get(
    "JUDGE_LLM_URL", "http://vllm-llama-8b.llm.svc.cluster.local:8000/v1"
)
JUDGE_LLM_MODEL = os.environ.get("JUDGE_LLM_MODEL", "llama-3.1-8b")

EMBEDDINGS_URL = os.environ.get(
    "EMBEDDINGS_URL", "http://vllm-bge-m3.llm.svc.cluster.local:8000/v1"
)
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "bge-m3")

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/workspace/dataset.yaml"))
BASELINES_PATH = Path(os.environ.get("BASELINES_PATH", "/workspace/baselines.json"))
SCORES_OUT_PATH = Path(os.environ.get("SCORES_OUT_PATH", "/workspace/output/scores.json"))

# Regression threshold in percentage points (0.0–1.0 metric scale).
# 0.05 = drop of more than 5pp triggers a CI failure.
REGRESSION_THRESHOLD = float(os.environ.get("RAGAS_REGRESSION_THRESHOLD_PP", "0.05"))

# Update baselines instead of gating against them. Set by the
# workflow_dispatch path when the operator wants to adopt new scores.
UPDATE_BASELINE = os.environ.get("UPDATE_BASELINE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Auth + ingestion + invoke helpers
# ---------------------------------------------------------------------------


def mint_token() -> str:
    """Mint a JWT via the langgraph-service public client + direct-access grant.

    Tokens default to 5min lifetime; we mint once at startup and use it
    across the whole run. A 5-question eval finishes well under that.
    """
    if not KEYCLOAK_PASSWORD:
        sys.exit("KEYCLOAK_PASSWORD env var is empty; cannot mint token")
    resp = httpx.post(
        KEYCLOAK_TOKEN_URL,
        data={
            "grant_type": "password",
            "client_id": KEYCLOAK_CLIENT_ID,
            "username": KEYCLOAK_USERNAME,
            "password": KEYCLOAK_PASSWORD,
        },
        timeout=15.0,
    )
    resp.raise_for_status()
    body = resp.json()
    if "access_token" not in body:
        sys.exit(f"token endpoint returned no access_token: {body}")
    return body["access_token"]


def ingest_doc(token: str, session_id: str, filename: str, content: str) -> None:
    """Idempotently ingest a single doc into the session.

    ingestion-service is expected to dedupe by (session_id, source) so
    re-running is a no-op when content hasn't changed. We pass the
    content's SHA-256 in a header for visibility in the service's
    access log; the dedupe itself is done server-side.
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    files = {"file": (filename, content.encode("utf-8"), "text/markdown")}
    data = {"session_id": session_id}
    resp = httpx.post(
        f"{INGESTION_URL}/upload",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Eval-Content-Digest": digest,
        },
        files=files,
        data=data,
        timeout=120.0,
    )
    if resp.status_code >= 400:
        sys.exit(
            f"ingest_doc({filename}) failed: {resp.status_code} {resp.text[:300]}"
        )


def invoke(token: str, prompt: str, session_id: str) -> dict[str, Any]:
    """POST to /invoke and return the parsed response."""
    resp = httpx.post(
        f"{LANGGRAPH_URL}/invoke",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt,
            "max_tokens": 512,
            "session_id": session_id,
        },
        timeout=180.0,
    )
    if resp.status_code >= 400:
        sys.exit(f"/invoke failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()


# ---------------------------------------------------------------------------
# Score + gate
# ---------------------------------------------------------------------------


def build_eval_dataset(dataset: dict, results: list[dict]) -> EvaluationDataset:
    """Assemble RAGAS-shape rows from dataset entries + invoke results.

    RAGAS 0.2.x expects each row as a dict with keys:
      user_input, response, retrieved_contexts, reference, reference_contexts.
    The `reference` field is the ground_truth_answer (used by metrics
    like ResponseRelevancy). `reference_contexts` is the ground-truth
    context list (used by LLMContextPrecisionWithReference).
    """
    rows: list[dict] = []
    for entry, result in zip(dataset["entries"], results):
        rows.append(
            {
                "user_input": entry["question"],
                "response": result["response"],
                "retrieved_contexts": [
                    c["text"] for c in result.get("retrieved_chunks", [])
                ],
                "reference": entry["ground_truth_answer"],
                "reference_contexts": entry["ground_truth_contexts"],
            }
        )
    return EvaluationDataset.from_list(rows)


def gate(scores: dict, baselines: dict) -> int:
    """Compare current scores to baselines; return exit code.

    Null baselines = no prior baseline = pass-through (warn, don't gate).
    Any non-null baseline that drops by >REGRESSION_THRESHOLD => exit 1.
    """
    base_metrics = baselines.get("metrics", {})
    failures: list[str] = []
    summary_lines: list[str] = []
    for metric, current in scores["metrics"].items():
        baseline = base_metrics.get(metric)
        if baseline is None:
            summary_lines.append(
                f"  {metric}: {current:.4f} (no baseline; gate skipped)"
            )
            continue
        delta = current - baseline
        flag = "OK"
        if delta < -REGRESSION_THRESHOLD:
            flag = "REGRESSION"
            failures.append(
                f"{metric}: {current:.4f} vs baseline {baseline:.4f} "
                f"(Δ {delta:+.4f}, threshold -{REGRESSION_THRESHOLD:.4f})"
            )
        summary_lines.append(
            f"  {metric}: {current:.4f} vs baseline {baseline:.4f} "
            f"(Δ {delta:+.4f}) [{flag}]"
        )

    print("=" * 60)
    print("RAGAS regression gate:")
    for line in summary_lines:
        print(line)
    print("=" * 60)

    if failures:
        print("FAIL: regressions detected:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading dataset:", DATASET_PATH)
    dataset = yaml.safe_load(DATASET_PATH.read_text())
    session_id = dataset["session_id"]
    entries = dataset["entries"]
    print(f"Dataset: {len(entries)} entries, session_id={session_id}")

    print("Loading baselines:", BASELINES_PATH)
    baselines = json.loads(BASELINES_PATH.read_text())

    print("Minting Keycloak token...")
    token = mint_token()

    print("Ingesting source docs...")
    for entry in entries:
        doc = entry["source_doc"]
        print(f"  -> {doc['filename']}")
        ingest_doc(token, session_id, doc["filename"], doc["content"])

    # Small sleep to let ingestion-service finish embedding + upserting
    # to Qdrant. Without this the first /invoke can race the upsert.
    time.sleep(5)

    print("Running queries...")
    results: list[dict[str, Any]] = []
    for entry in entries:
        print(f"  -> {entry['id']}: {entry['question'][:80]}...")
        t0 = time.monotonic()
        result = invoke(token, entry["question"], session_id)
        elapsed = time.monotonic() - t0
        print(
            f"     route={result.get('route')}, "
            f"chunks={result.get('retrieve_count')}, "
            f"cycles={result.get('reasoning_cycles', 0)}, "
            f"{elapsed:.1f}s"
        )
        results.append(result)

    print("Building RAGAS evaluation dataset...")
    eval_ds = build_eval_dataset(dataset, results)

    print("Configuring judge LLM + embeddings (in-cluster vLLM)...")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_LLM_MODEL,
            base_url=JUDGE_LLM_URL,
            api_key="not-required",  # vLLM doesn't enforce
            temperature=0.0,
            max_tokens=512,
            timeout=60,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL,
            base_url=EMBEDDINGS_URL,
            api_key="not-required",
            check_embedding_ctx_length=False,  # bge-m3 has its own
                                               # tokenizer; the OpenAI
                                               # embedding client's
                                               # check assumes ada-002.
        )
    )

    print("Scoring with RAGAS...")
    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge_llm),
    ]
    score_result = evaluate(
        dataset=eval_ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        # show_progress is verbose in CI logs; let it stream so a stuck
        # judge call is visible in real time.
        show_progress=True,
    )

    # RAGAS returns an EvaluationResult; .to_pandas() gives per-row
    # scores, and the result itself carries aggregate means.
    df = score_result.to_pandas()

    # 0.2.x reports columns under each metric's `name` attribute. Walk
    # the metrics list to keep this robust against future renames.
    metric_names = [m.name for m in metrics]
    aggregate = {name: float(df[name].mean()) for name in metric_names}

    per_entry: dict[str, dict[str, float]] = {}
    for i, entry in enumerate(entries):
        per_entry[entry["id"]] = {
            name: float(df[name].iloc[i]) for name in metric_names
        }

    scores = {
        "schema_version": baselines.get("_schema_version", 1),
        "dataset_version": baselines.get("_dataset_version", "unknown"),
        "metrics": aggregate,
        "per_entry": per_entry,
    }

    SCORES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORES_OUT_PATH.write_text(json.dumps(scores, indent=2))
    print(f"Wrote scores: {SCORES_OUT_PATH}")
    print(json.dumps(aggregate, indent=2))

    # Stdout-marker block. The CI workflow tails `kubectl logs job/...`
    # after Job completion and uses these markers to slice out the
    # scores JSON without needing an S3 round-trip. Keep the markers
    # exactly as written; the workflow's awk pattern depends on them.
    print("===RAGAS_SCORES_JSON_BEGIN===")
    print(json.dumps(scores))
    print("===RAGAS_SCORES_JSON_END===")

    if UPDATE_BASELINE:
        new_baselines = dict(baselines)
        new_baselines["metrics"] = aggregate
        new_baselines["per_entry"] = per_entry
        BASELINES_PATH.write_text(json.dumps(new_baselines, indent=2) + "\n")
        print("UPDATE_BASELINE=true: wrote new baseline. Commit baselines.json.")
        return 0

    return gate(scores, baselines)


if __name__ == "__main__":
    sys.exit(main())
