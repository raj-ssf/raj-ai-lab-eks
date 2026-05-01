#!/usr/bin/env python3
"""Phase #81b: Generate teacher-output dataset for distillation.

Takes a JSONL file of prompts (one per line, each with at least a
"prompt" field), runs each prompt through a vLLM teacher endpoint
via the OpenAI chat-completions API, and writes (instruction, input,
output) tuples in alpaca-style JSONL ready for Axolotl SFT
(training/configs/llama-3.2-1b-distilled.yaml expects this format).

Output is uploaded to S3 at TEACHER_OUTPUT_S3_URI (read by the
training PyTorchJob's dataset-sync init container).

Failure modes:
  - vLLM teacher OOM / 429: skip the prompt with a logged warning,
    continue. A small fraction of skipped prompts is acceptable for
    distillation; the dataset is large.
  - Teacher pod restarts mid-run: SDK retries built into openai
    library handle transient. Hard failures: re-run.
  - S3 upload failure at end: the local JSONL is preserved at
    /tmp/teacher_outputs.jsonl. Re-run the upload manually.

Concurrency:
  Sequential by default for clarity. Set MAX_CONCURRENT > 1 to
  parallelize requests (vLLM's continuous-batching scheduler
  handles concurrent decodes natively, so 16-32 in-flight
  shouldn't stress the teacher).
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("seed_prompts")


def load_prompts(prompts_path: Path) -> list[str]:
    """Load prompts from a JSONL file. Each line should have a 'prompt' or
    'question' or 'instruction' field; we accept any of those for
    flexibility (the lab's eval datasets use 'question').
    """
    prompts = []
    with prompts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("prompt") or obj.get("question") or obj.get("instruction")
            if not text:
                log.warning("skipping line missing prompt/question/instruction field: %s", obj)
                continue
            prompts.append(text)
    log.info("loaded %d prompts from %s", len(prompts), prompts_path)
    return prompts


def call_teacher(client: OpenAI, model: str, prompt: str, max_tokens: int) -> str:
    """One teacher inference. Returns the assistant's response text.
    Raises on any non-recoverable error (caller decides whether to
    skip vs abort).
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        # temperature=0 for reproducibility — we want deterministic
        # teacher outputs so re-running the dataset prep produces
        # an identical training set.
        temperature=0.0,
    )
    return resp.choices[0].message.content


def process_one(client: OpenAI, model: str, prompt: str, max_tokens: int) -> dict | None:
    """Wrap call_teacher with retry + skip-on-failure semantics. Returns
    an alpaca-format dict ready for JSONL serialization, or None if
    the prompt was unrecoverable.
    """
    for attempt in range(3):
        try:
            output = call_teacher(client, model, prompt, max_tokens)
            return {"instruction": prompt, "input": "", "output": output}
        except Exception as e:
            log.warning("attempt %d failed for prompt[:60]=%r: %s", attempt + 1, prompt[:60], e)
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    log.error("giving up on prompt: %s", prompt[:80])
    return None


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    """Upload local file to s3://bucket/key using boto3. Pod Identity
    creds are auto-discovered (no explicit creds needed).
    """
    assert s3_uri.startswith("s3://"), f"expected s3:// URI, got {s3_uri}"
    bucket, key = s3_uri[len("s3://"):].split("/", 1)
    s3 = boto3.client("s3")
    log.info("uploading %s → s3://%s/%s", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)
    log.info("upload complete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-file", required=True, help="JSONL file of prompts")
    p.add_argument("--output-file", default="/tmp/teacher_outputs.jsonl")
    p.add_argument("--output-s3-uri", required=True, help="s3://bucket/key for upload")
    p.add_argument("--teacher-url", required=True, help="e.g. http://vllm.llm.svc.cluster.local:8000/v1")
    p.add_argument("--teacher-model", required=True, help="e.g. llama-3.3-70b")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="parallel in-flight requests; vLLM handles 30+ comfortably")
    args = p.parse_args()

    prompts = load_prompts(Path(args.prompts_file))
    if not prompts:
        log.error("no prompts loaded; aborting")
        sys.exit(1)

    # vLLM exposes an OpenAI-compatible endpoint; api_key is unused
    # by vllm but the SDK requires a non-empty value.
    client = OpenAI(base_url=args.teacher_url, api_key="not-required", timeout=60.0)

    log.info("starting teacher inference: %d prompts × max_tokens=%d, concurrency=%d",
             len(prompts), args.max_tokens, args.max_concurrent)
    start = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = [pool.submit(process_one, client, args.teacher_model, prompt, args.max_tokens)
                   for prompt in prompts]
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result is not None:
                results.append(result)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                log.info("progress: %d/%d (%.1fs elapsed, ~%.1fs/prompt)",
                         i + 1, len(prompts), elapsed, elapsed / (i + 1))

    log.info("teacher inference complete: %d/%d prompts succeeded in %.1fs",
             len(results), len(prompts), time.time() - start)

    # Write JSONL — order by original prompt index for reproducibility.
    # ThreadPoolExecutor's as_completed is fastest-first; sort to fix.
    # (Map back via prompt match — could keep an index but this is simpler
    # given prompts are unique in our dataset.)
    prompt_to_result = {r["instruction"]: r for r in results}
    out_path = Path(args.output_file)
    with out_path.open("w") as f:
        for prompt in prompts:
            r = prompt_to_result.get(prompt)
            if r is not None:
                f.write(json.dumps(r) + "\n")
    log.info("wrote %d lines to %s", len(results), out_path)

    upload_to_s3(out_path, args.output_s3_uri)


if __name__ == "__main__":
    main()
