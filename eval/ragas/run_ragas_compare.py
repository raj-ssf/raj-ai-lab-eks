#!/usr/bin/env python3
"""
Phase #81f: side-by-side RAGAS quality comparison between the 8B
trivial-tier baseline (vllm-llama-8b serving llama-3.1-8b) and the
1B distilled candidate (vllm-llama-1b-distilled serving the
llama-3.2-1b-distilled adapter).

Pipeline:
  1. Load eval/ragas/dataset.yaml — 5 questions with ground-truth
     answers + ground-truth contexts.
  2. For each question, build the SAME user prompt for both models
     (question + ground-truth contexts folded in). This deliberately
     SKIPS retrieval to remove rag-service variance from the
     comparison — pure question-answering quality.
  3. POST to each model's /v1/chat/completions. Each model applies
     its own chat template (Llama-3.1-Instruct's bundled template
     for 8B, the Phase #81d Alpaca template for 1B-distilled).
  4. Score both result-sets with RAGAS faithfulness +
     answer_relevancy. Skip context_precision (it scores retrieval,
     which we bypassed).
  5. Print a side-by-side table; emit a JSON report.

Why this design vs going through langgraph-service:
  langgraph adds classifier routing, planner, RAG retrieval, and
  tool injection — all useful for end-to-end testing but they
  introduce variance in BOTH paths and make it hard to isolate
  "is the 1B inference quality good enough?" This script answers
  THAT question directly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_user_prompt(question: str, contexts: list[str]) -> str:
    """Pack the RAG context + question into a single user message.

    Both models will see the same string. Each applies its own chat
    template on top.
    """
    ctx_block = "\n\n".join(f"- {c}" for c in contexts)
    return (
        "Answer the question using ONLY the context below. If the "
        "context doesn't contain the answer, say so directly.\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {question}"
    )


def call_model(url: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    """One /v1/chat/completions call. Returns assistant text."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=180.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"unexpected response from {model}: {data}")
    return data["choices"][0]["message"]["content"]


def score_run(
    rows: list[dict],
    judge_url: str,
    judge_model: str,
    embeddings_url: str,
    embeddings_model: str,
) -> dict:
    """Score a list of (question, response, contexts, ground_truth) rows.

    Returns aggregate + per-entry RAGAS scores.
    """
    eval_ds = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["response"],
                "retrieved_contexts": r["contexts"],
                "reference": r["ground_truth"],
            }
            for r in rows
        ]
    )
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            base_url=judge_url,
            api_key="not-required",
            temperature=0.0,
            max_tokens=512,
            timeout=120,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=embeddings_model,
            base_url=embeddings_url,
            api_key="not-required",
            check_embedding_ctx_length=False,
        )
    )
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
    ]
    res = evaluate(
        dataset=eval_ds,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        show_progress=False,
    )
    df = res.to_pandas()
    names = [m.name for m in metrics]
    aggregate = {n: float(df[n].mean()) for n in names}
    per_entry = [
        {"id": rows[i]["id"], **{n: float(df[n].iloc[i]) for n in names}}
        for i in range(len(rows))
    ]
    return {"aggregate": aggregate, "per_entry": per_entry}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eval/ragas/dataset.yaml")
    # Generation URLs (call_model appends /v1/chat/completions itself).
    p.add_argument("--baseline-url", default="http://localhost:18002")  # 8B
    p.add_argument("--baseline-model", default="llama-3.1-8b")
    p.add_argument("--candidate-url", default="http://localhost:18001")  # 1B
    p.add_argument("--candidate-model", default="llama-3.2-1b-distilled")
    # Judge / embeddings URLs MUST include /v1 — langchain_openai's
    # ChatOpenAI/OpenAIEmbeddings call <base_url>/chat/completions
    # and <base_url>/embeddings, expecting base_url to be the
    # OpenAI-compatible /v1 root. Without /v1 you get HTTP 404 from
    # vLLM (which serves at /v1/chat/completions, not /chat/completions).
    p.add_argument("--judge-url", default="http://localhost:18002/v1")
    p.add_argument("--judge-model", default="llama-3.1-8b")
    p.add_argument("--embeddings-url", default="http://localhost:18003/v1")
    p.add_argument("--embeddings-model", default="bge-m3")
    p.add_argument("--output", default="/tmp/ragas-compare-scores.json")
    args = p.parse_args()

    ds = yaml.safe_load(Path(args.dataset).read_text())
    entries = ds["entries"]
    print(f"Dataset: {len(entries)} entries from {args.dataset}")

    # --- Phase A: generate responses from both models ---
    print(f"\n=== Generating responses ===")
    rows_baseline: list[dict] = []
    rows_candidate: list[dict] = []
    for e in entries:
        prompt = build_user_prompt(e["question"], e["ground_truth_contexts"])
        print(f"  {e['id']}: prompt_len={len(prompt)} chars")

        t0 = time.time()
        resp_b = call_model(args.baseline_url, args.baseline_model, prompt)
        t_b = time.time() - t0
        print(f"    baseline ({args.baseline_model}): {t_b:.1f}s, {len(resp_b)} chars")

        t0 = time.time()
        resp_c = call_model(args.candidate_url, args.candidate_model, prompt)
        t_c = time.time() - t0
        print(f"    candidate ({args.candidate_model}): {t_c:.1f}s, {len(resp_c)} chars")

        common = {
            "id": e["id"],
            "question": e["question"],
            "contexts": e["ground_truth_contexts"],
            "ground_truth": e["ground_truth_answer"],
        }
        rows_baseline.append({**common, "response": resp_b, "latency_s": t_b})
        rows_candidate.append({**common, "response": resp_c, "latency_s": t_c})

    # --- Phase B: RAGAS scoring (judge LLM = 8B) ---
    print(f"\n=== Scoring baseline ({args.baseline_model}) ===")
    scores_b = score_run(
        rows_baseline,
        args.judge_url,
        args.judge_model,
        args.embeddings_url,
        args.embeddings_model,
    )
    print(json.dumps(scores_b["aggregate"], indent=2))

    print(f"\n=== Scoring candidate ({args.candidate_model}) ===")
    scores_c = score_run(
        rows_candidate,
        args.judge_url,
        args.judge_model,
        args.embeddings_url,
        args.embeddings_model,
    )
    print(json.dumps(scores_c["aggregate"], indent=2))

    # --- Phase C: side-by-side report ---
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<22} {args.baseline_model:>20} {args.candidate_model:>20}  Δ")
    print("-" * 70)
    for metric in scores_b["aggregate"].keys():
        b = scores_b["aggregate"][metric]
        c = scores_c["aggregate"][metric]
        delta = c - b
        print(f"{metric:<22} {b:>20.4f} {c:>20.4f}  {delta:+.4f}")

    # Per-entry comparison for diagnostic visibility
    print(f"\n{'=' * 70}")
    print("Per-entry latency:")
    for rb, rc in zip(rows_baseline, rows_candidate):
        print(
            f"  {rb['id']:<28} "
            f"baseline={rb['latency_s']:5.1f}s  "
            f"candidate={rc['latency_s']:5.1f}s  "
            f"speedup={rb['latency_s']/rc['latency_s']:4.2f}x"
        )

    output = {
        "baseline": {
            "url": args.baseline_url,
            "model": args.baseline_model,
            **scores_b,
            "responses": [
                {"id": r["id"], "response": r["response"], "latency_s": r["latency_s"]}
                for r in rows_baseline
            ],
        },
        "candidate": {
            "url": args.candidate_url,
            "model": args.candidate_model,
            **scores_c,
            "responses": [
                {"id": r["id"], "response": r["response"], "latency_s": r["latency_s"]}
                for r in rows_candidate
            ],
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nFull results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
