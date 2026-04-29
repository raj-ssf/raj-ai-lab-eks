#!/usr/bin/env python3
"""
Embedding drift detector for vllm-bge-m3.

For each frozen canary snippet in dataset.yaml, embed via the live
vllm-bge-m3 endpoint, compute cosine distance vs the baseline
embedding stored in baselines.json. Flags drift when:
  - Any single snippet's distance exceeds DRIFT_PER_SNIPPET_THRESHOLD
    (default 0.05 — 5% from baseline). Catches localized drift in
    one semantic region.
  - Mean drift across all snippets exceeds DRIFT_MEAN_THRESHOLD
    (default 0.02). Catches global model shifts that move every
    embedding slightly.

UPDATE_BASELINE=true mode: skips comparison, writes the current
embeddings as the new baseline. Operator runs once on a known-good
cluster state, then commits baselines.json. Future runs compare
against that snapshot.

Why this matters: vllm-bge-m3 image bumps, GPU driver changes, or
vLLM internals can subtly shift the embedding space landscape. When
that happens, pre-existing cached embeddings + Qdrant chunks are
LESS aligned with new queries — silent retrieval quality regression.
This Job catches the shift before users notice.

Failure mode: Same as RAGAS runner — every step has a single retry,
hard failures cause exit(1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml


# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------

EMBEDDINGS_URL = os.environ.get(
    "EMBEDDINGS_URL", "http://vllm-bge-m3.llm.svc.cluster.local:8000/v1"
)
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "bge-m3")

DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/eval/dataset.yaml"))
BASELINES_PATH = Path(os.environ.get("BASELINES_PATH", "/eval/baselines.json"))
SCORES_OUT_PATH = Path(
    os.environ.get("SCORES_OUT_PATH", "/workspace/drift-scores.json")
)

# Per-snippet drift threshold. Cosine DISTANCE (1 - cosine similarity)
# range is [0, 2]; for normalized embeddings it's typically [0, ~0.3]
# even for unrelated texts, so 0.05 is a meaningful "shift detected"
# bound.
DRIFT_PER_SNIPPET_THRESHOLD = float(
    os.environ.get("DRIFT_PER_SNIPPET_THRESHOLD", "0.05")
)
# Mean drift threshold. Catches the case where every snippet drifted
# slightly (global model shift) without any single snippet exceeding
# the per-snippet bound.
DRIFT_MEAN_THRESHOLD = float(os.environ.get("DRIFT_MEAN_THRESHOLD", "0.02"))

UPDATE_BASELINE = os.environ.get("UPDATE_BASELINE", "").lower() in (
    "1",
    "true",
    "yes",
)

EMBED_TIMEOUT_SECONDS = float(os.environ.get("EMBED_TIMEOUT_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Call /v1/embeddings on the configured model. Raises on error
    so the runner exits non-zero — drift detection isn't useful when
    the embedding endpoint itself is broken."""
    with httpx.Client(timeout=EMBED_TIMEOUT_SECONDS) as client:
        resp = client.post(
            f"{EMBEDDINGS_URL}/embeddings",
            json={"model": EMBEDDINGS_MODEL, "input": text},
        )
        resp.raise_for_status()
        body = resp.json()
        return body["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
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


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity. Range [0, 2]; for normalized
    embeddings typically [0, ~0.3]."""
    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading dataset:", DATASET_PATH)
    dataset = yaml.safe_load(DATASET_PATH.read_text())
    snippets = dataset["snippets"]
    print(f"Dataset: {len(snippets)} canary snippets")

    print("Loading baselines:", BASELINES_PATH)
    baselines = json.loads(BASELINES_PATH.read_text())
    baseline_embeddings: dict[str, list[float]] = baselines.get("embeddings", {}) or {}

    print(f"Embedding {len(snippets)} snippets via {EMBEDDINGS_MODEL} ...")
    current_embeddings: dict[str, list[float]] = {}
    for snippet in snippets:
        sid = snippet["id"]
        text = snippet["text"]
        t0 = time.monotonic()
        emb = embed_text(text)
        elapsed = time.monotonic() - t0
        current_embeddings[sid] = emb
        print(f"  -> {sid:20s} dim={len(emb)} ({elapsed:.2f}s)")

    if UPDATE_BASELINE:
        new_baselines = dict(baselines)
        new_baselines["embeddings"] = current_embeddings
        new_baselines["model_at_baseline"] = EMBEDDINGS_MODEL
        new_baselines["captured_at"] = datetime.now(timezone.utc).isoformat()
        BASELINES_PATH.write_text(json.dumps(new_baselines, indent=2) + "\n")
        print(
            f"\nUPDATE_BASELINE=true: wrote {len(current_embeddings)} "
            f"baseline embeddings."
        )
        print("Commit baselines.json. Subsequent runs compare against this snapshot.")
        return 0

    if not baseline_embeddings:
        # First-run mode — no baseline yet. Emit scores but skip gate.
        print(
            "\nNo baseline yet (baselines.json.embeddings is empty). "
            "Run with UPDATE_BASELINE=true on a known-good state to "
            "populate. Skipping gate this run."
        )
        scores: dict[str, Any] = {
            "schema_version": 1,
            "model": EMBEDDINGS_MODEL,
            "snippet_count": len(snippets),
            "baseline_present": False,
            "per_snippet_distance": {},
            "max_distance": 0.0,
            "mean_distance": 0.0,
        }
        SCORES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORES_OUT_PATH.write_text(json.dumps(scores, indent=2))
        return 0

    # Compute drift per snippet
    per_snippet_distance: dict[str, float] = {}
    for sid, emb in current_embeddings.items():
        baseline = baseline_embeddings.get(sid)
        if baseline is None:
            print(f"  WARN: no baseline for {sid} — added to dataset since baseline was captured. Skipping.")
            continue
        d = cosine_distance(emb, baseline)
        per_snippet_distance[sid] = d

    if not per_snippet_distance:
        print(
            "\nNo overlap between current snippets and baseline "
            "embeddings — dataset evolved completely since baseline "
            "was captured. Run UPDATE_BASELINE=true to refresh."
        )
        return 1

    distances = list(per_snippet_distance.values())
    max_d = max(distances)
    mean_d = sum(distances) / len(distances)

    print("\n" + "=" * 70)
    print("Drift report (cosine distance vs baseline; 0 = identical):")
    for sid, d in sorted(per_snippet_distance.items(), key=lambda x: -x[1]):
        flag = ""
        if d > DRIFT_PER_SNIPPET_THRESHOLD:
            flag = " ← DRIFT"
        print(f"  {sid:20s} {d:.6f}{flag}")
    print("-" * 70)
    print(f"  max:  {max_d:.6f}  (threshold {DRIFT_PER_SNIPPET_THRESHOLD})")
    print(f"  mean: {mean_d:.6f}  (threshold {DRIFT_MEAN_THRESHOLD})")
    print("=" * 70)

    scores = {
        "schema_version": 1,
        "model": EMBEDDINGS_MODEL,
        "baseline_model": baselines.get("model_at_baseline"),
        "baseline_captured_at": baselines.get("captured_at"),
        "snippet_count": len(per_snippet_distance),
        "per_snippet_distance": per_snippet_distance,
        "max_distance": max_d,
        "mean_distance": mean_d,
        "per_snippet_threshold": DRIFT_PER_SNIPPET_THRESHOLD,
        "mean_threshold": DRIFT_MEAN_THRESHOLD,
    }
    SCORES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORES_OUT_PATH.write_text(json.dumps(scores, indent=2))

    # Stdout markers for log-scrape extraction (same pattern as RAGAS).
    print("===DRIFT_SCORES_JSON_BEGIN===")
    print(json.dumps(scores))
    print("===DRIFT_SCORES_JSON_END===")

    failures: list[str] = []
    if max_d > DRIFT_PER_SNIPPET_THRESHOLD:
        worst = max(per_snippet_distance.items(), key=lambda x: x[1])
        failures.append(
            f"per-snippet drift: {worst[0]} = {worst[1]:.6f} > "
            f"{DRIFT_PER_SNIPPET_THRESHOLD}"
        )
    if mean_d > DRIFT_MEAN_THRESHOLD:
        failures.append(
            f"mean drift: {mean_d:.6f} > {DRIFT_MEAN_THRESHOLD}"
        )

    if failures:
        print("\nFAIL: embedding drift detected.")
        for f in failures:
            print(f"  - {f}")
        print(
            "\nRecommended actions:\n"
            "  1. Investigate what changed (model version, GPU driver,\n"
            "     vLLM image, etc.) — git log on llm/base/ + ECR\n"
            "     image tags is a starting point.\n"
            "  2. If drift is intentional (model upgrade), re-run with\n"
            "     UPDATE_BASELINE=true and consider re-embedding any\n"
            "     persisted Qdrant chunks (their embeddings are now\n"
            "     out-of-distribution vs new queries).\n"
            "  3. If drift is unexpected, roll back the offending change."
        )
        return 1

    print("\nPASS: no drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
