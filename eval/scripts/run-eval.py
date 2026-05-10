#!/usr/bin/env python3
"""
Parallel base-vs-LoRA eval runner.

Runs lm-evaluation-harness twice concurrently — once against the base
model name (`llama-3.1-8b`) and once against the LoRA-merged name
(`llama-3.1-8b-alpaca`) — both pointed at the same in-cluster vLLM
Service. vLLM batches requests across both flights on the same GPU, so
running them concurrently is faster than sequential AND demonstrates
that the LoRA-tuned model is genuinely a different inference path
(otherwise the request batcher would see them as identical and we'd
get suspicious tied scores).

After both eval runs complete:
  1. Parses the per-task JSON results files lm-eval writes
  2. Prints a side-by-side summary to stdout
  3. Optionally emits a Langfuse trace (one trace per run, with per-task
     spans) — only if LANGFUSE_PUBLIC_KEY env is set. Otherwise no-op.
  4. `aws s3 sync`s the entire output dir to s3://...eval-results/<run-id>/

Configurable via env vars (all optional except the defaults make sense
for the F4 smoke):
  VLLM_URL          — base URL of the vLLM OpenAI-compatible server
                      (default http://vllm-llama-8b:8000/v1)
  TASKS             — comma-separated lm-eval task names
                      (default mmlu_anatomy,arc_easy,hellaswag — small
                      smoke set; full mmlu would be ~4 hr)
  LIMIT             — per-task sample cap (default 100)
  S3_DEST           — destination prefix for results JSON
                      (default s3://raj-ai-lab-eks-cilium-training/eval-results/<run-id>)
  RUN_ID            — short tag for this run (default = ISO timestamp)
  LANGFUSE_HOST     — Langfuse API URL
                      (default http://langfuse-web.langfuse.svc.cluster.local:3000)
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY — Langfuse SDK creds.
                      If unset, Langfuse trace emission is skipped.
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

VLLM_URL    = os.environ.get("VLLM_URL", "http://vllm-llama-8b:8000/v1")
TASKS       = os.environ.get("TASKS", "mmlu_anatomy,arc_easy,hellaswag")
LIMIT       = os.environ.get("LIMIT", "100")
RUN_ID      = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
S3_DEST     = os.environ.get("S3_DEST") or f"s3://raj-ai-lab-eks-cilium-training/eval-results/{RUN_ID}"

# HF Hub tokenizer to use locally for prompt construction + logprob math.
# The lm-eval `local-completions` adapter ALWAYS loads a tokenizer to
# format multiple-choice options for logprob comparison (MMLU, HellaSwag,
# ARC are all logprob-based tasks). It defaults to using the model NAME
# as the tokenizer identifier — but our vLLM `served-model-name`
# (`llama-3.1-8b`) is an arbitrary label, not an HF Hub model ID.
# So we explicitly point the tokenizer at the open Llama 3.1 mirror.
# Both base and LoRA flights use the same tokenizer (LoRA doesn't touch
# tokenization).
TOKENIZER   = os.environ.get("TOKENIZER", "NousResearch/Meta-Llama-3.1-8B")

OUTPUT_DIR  = Path("/workspace/output") / RUN_ID
MODELS = [
    ("llama-3.1-8b",        OUTPUT_DIR / "base"),
    ("llama-3.1-8b-alpaca", OUTPUT_DIR / "lora"),
]

# ---------------------------------------------------------------------------
# Run lm-eval for one model in a thread
# ---------------------------------------------------------------------------

def run_eval(model_name: str, output_path: Path) -> int:
    """Invoke lm_eval against vLLM for the given model name. Returns exit code."""
    output_path.mkdir(parents=True, exist_ok=True)

    # local-completions: lm-eval's adapter for an OpenAI-compatible
    # server hosted locally (or in-cluster). num_concurrent batches
    # parallel requests at the HTTP-client side; vLLM also batches
    # server-side so total throughput is ~num_concurrent * batch_size.
    # tokenized_requests=False uses string prompts (not pre-tokenized
    # token-ids) — required when the eval client doesn't have a
    # matching tokenizer locally.
    model_args = (
        f"model={model_name},"
        f"base_url={VLLM_URL}/completions,"
        f"tokenizer={TOKENIZER},"
        "num_concurrent=4,"
        "max_retries=3,"
        "tokenized_requests=False"
    )

    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", TASKS,
        "--limit", LIMIT,
        "--output_path", str(output_path),
        "--write_out",  # one JSON per task with prompts + responses
    ]

    print(f"[{model_name}] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=False)
    print(f"[{model_name}] exit code: {proc.returncode}", flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# Parallel run + summary
# ---------------------------------------------------------------------------

def main():
    print(f"=== Eval run {RUN_ID} ===")
    print(f"vLLM:    {VLLM_URL}")
    print(f"Tasks:   {TASKS}")
    print(f"Limit:   {LIMIT} per task")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"S3:      {S3_DEST}")
    print()

    # Threads (one per model). Both run concurrently — vLLM auto-batches
    # incoming requests from both threads on the same GPU.
    results = {}
    threads = []

    def runner(model_name, path):
        rc = run_eval(model_name, path)
        results[model_name] = {"exit_code": rc, "path": path}

    for model, path in MODELS:
        t = threading.Thread(target=runner, args=(model, path), name=f"eval-{model}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Failure: bail before S3/Langfuse to keep error obvious in pod logs.
    failed = [m for m, r in results.items() if r["exit_code"] != 0]
    if failed:
        print(f"ERROR: eval failed for: {failed}", file=sys.stderr)
        # Still try to sync partial output to S3 for post-mortem.
        s3_sync()
        sys.exit(1)

    # Parse the results JSON each lm-eval run wrote.
    # lm-eval writes a results_<timestamp>.json at the output_path root.
    parsed = {}
    for model, info in results.items():
        json_files = sorted(info["path"].rglob("results_*.json"))
        if not json_files:
            print(f"WARN: no results JSON found for {model} under {info['path']}")
            continue
        with json_files[-1].open() as f:
            data = json.load(f)
        parsed[model] = data.get("results", {})

    print_summary(parsed)
    emit_langfuse_trace(parsed)
    s3_sync()
    print(f"\n=== Done. Results at {S3_DEST}/ ===")


def print_summary(parsed):
    print("\n" + "=" * 70)
    print("=== SUMMARY ===")
    print("=" * 70)
    if not parsed:
        print("No results parsed.")
        return
    # Build per-task table: rows = tasks, cols = models
    tasks = sorted({t for r in parsed.values() for t in r.keys()})
    models = list(parsed.keys())
    header = f"{'task':<32}" + "".join(f"{m:<28}" for m in models)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = f"{task:<32}"
        for m in models:
            metrics = parsed[m].get(task, {})
            # Pick the primary metric: acc, acc_norm, exact_match, etc.
            for key in ("acc,none", "acc_norm,none", "exact_match,none", "acc", "acc_norm"):
                if key in metrics:
                    row += f"{metrics[key]:<28.4f}"
                    break
            else:
                row += f"{'(no metric)':<28}"
        print(row)


def emit_langfuse_trace(parsed):
    """Push a single trace summarizing both runs. No-op if creds unset."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        print("\n[langfuse] LANGFUSE_PUBLIC_KEY not set — skipping trace.")
        return
    try:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000"),
        )
        trace = lf.trace(
            name="alpaca-vs-base-eval",
            metadata={
                "run_id": RUN_ID,
                "vllm_url": VLLM_URL,
                "tasks": TASKS,
                "limit": LIMIT,
            },
        )
        for model, results in parsed.items():
            trace.span(
                name=f"eval:{model}",
                metadata={"model": model, "results": results},
            )
        lf.flush()
        print(f"\n[langfuse] Trace emitted (run_id={RUN_ID})")
    except Exception as e:
        # Don't fail the whole eval over an observability problem.
        print(f"\n[langfuse] WARN: trace emission failed: {e}", file=sys.stderr)


def s3_sync():
    """Sync the output dir to S3. Pod Identity supplies AWS creds."""
    cmd = ["aws", "s3", "sync", str(OUTPUT_DIR), S3_DEST, "--no-progress"]
    print(f"\n[s3] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
