"""
MT-Bench eval orchestrator: gen answers from TARGET + BASELINE, judge
pairwise via JUDGE_MODEL, score per category, compare to baselines,
upload results to S3, exit non-zero on regression.

Reads from env (see eval/runs/mtbench-eval.yaml). All HTTP calls are
to vLLM's OpenAI-compatible /v1/chat/completions endpoint.

This script is intentionally framework-light: httpx, pyyaml, boto3.
No FastChat / lm-eval / langchain deps — keeps the runtime image
to python:3.11-slim + pip install (~12s) per the Job spec.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import pathlib
import re
import sys
import typing as t

import boto3
import httpx
import yaml


# ---------------------------------------------------------------------------
# Config (from env)
# ---------------------------------------------------------------------------

TARGET_MODEL    = os.environ["TARGET_MODEL"]
BASELINE_MODEL  = os.environ["BASELINE_MODEL"]
VLLM_URL        = os.environ["VLLM_URL"].rstrip("/")
JUDGE_MODEL     = os.environ["JUDGE_MODEL"]
JUDGE_URL       = os.environ["JUDGE_URL"].rstrip("/")
JUDGE_PROVIDER  = os.environ.get("JUDGE_PROVIDER", "local")   # local | openai
SCORE_MODE      = os.environ.get("SCORE_MODE", "pairwise")    # pairwise | single
CATEGORIES      = os.environ.get("CATEGORIES", "all")
RESULTS_BUCKET  = os.environ["RESULTS_BUCKET"]
RUN_ID          = os.environ.get("RUN_ID") or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
THRESHOLD_PCT   = float(os.environ.get("REGRESSION_THRESHOLD_PCT", "5"))

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")  # only for JUDGE_PROVIDER=openai

WORKDIR         = pathlib.Path("/workspace")
QUESTIONS_PATH  = WORKDIR / "questions.yaml"
BASELINES_PATH  = WORKDIR / "baselines.json"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

class Question(t.TypedDict):
    id: int
    category: str           # writing | roleplay | reasoning | math | coding |
                            # extraction | stem | humanities
    turns: list[str]        # 1 or 2 user turns (MT-Bench is multi-turn)


class Answer(t.TypedDict):
    question_id: int
    model: str
    turns: list[str]        # model's response per user turn


class Verdict(t.TypedDict):
    question_id: int
    category: str
    winner: str             # "target" | "baseline" | "tie"
    judge_rationale: str


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_questions() -> list[Question]:
    with QUESTIONS_PATH.open() as f:
        all_qs = yaml.safe_load(f)["questions"]
    if CATEGORIES != "all":
        wanted = {c.strip() for c in CATEGORIES.split(",")}
        all_qs = [q for q in all_qs if q["category"] in wanted]
    print(f"Loaded {len(all_qs)} questions across categories: "
          f"{sorted({q['category'] for q in all_qs})}", flush=True)
    return all_qs


def load_baselines() -> dict[str, float]:
    if not BASELINES_PATH.exists():
        print("No baselines.json yet — first run; nothing to compare against.")
        return {}
    return json.loads(BASELINES_PATH.read_text())


# ---------------------------------------------------------------------------
# Generation (calls vLLM /v1/chat/completions per turn)
# ---------------------------------------------------------------------------

def gen_answer(client: httpx.Client, model: str, question: Question) -> Answer:
    """Run the multi-turn conversation, capturing each turn's response."""
    messages: list[dict] = []
    responses: list[str] = []
    for user_turn in question["turns"]:
        messages.append({"role": "user", "content": user_turn})
        resp = client.post(
            f"{VLLM_URL}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        responses.append(content)
        messages.append({"role": "assistant", "content": content})
    return {"question_id": question["id"], "model": model, "turns": responses}


def gen_all_answers(questions: list[Question]) -> tuple[list[Answer], list[Answer]]:
    """Generate answers from both models. Sequential (vLLM batches
    internally; the per-question latency hit is small)."""
    target, baseline = [], []
    with httpx.Client() as client:
        for q in questions:
            print(f"  [target]   q{q['id']:>3} ({q['category']}) ...", flush=True)
            target.append(gen_answer(client, TARGET_MODEL, q))
            print(f"  [baseline] q{q['id']:>3} ({q['category']}) ...", flush=True)
            baseline.append(gen_answer(client, BASELINE_MODEL, q))
    return target, baseline


# ---------------------------------------------------------------------------
# Judging (pairwise comparison via the judge model)
# ---------------------------------------------------------------------------

PAIRWISE_JUDGE_PROMPT = """You are an impartial judge evaluating two AI assistant answers \
to a user question. Compare the two answers and decide which is better, or call \
them a tie. Consider helpfulness, accuracy, depth, and instruction-following. \
Position bias is NOT acceptable — if the answers are equivalent in quality, \
respond "C" (tie).

[User Question]
{question}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

After your reasoning, output a single line starting with "Verdict: " followed \
by exactly one of: A, B, or C.

Verdict A means Answer A is better. Verdict B means Answer B is better. \
Verdict C means tie.
"""


def judge_pairwise(
    client: httpx.Client,
    question: Question,
    target_ans: Answer,
    baseline_ans: Answer,
    swap: bool,
) -> Verdict:
    """One judge call. swap=True puts target as B (vs A) — we run both
    orderings to cancel positional bias, then average."""
    a_ans, b_ans = (
        (baseline_ans, target_ans) if swap
        else (target_ans, baseline_ans)
    )
    # Concatenate multi-turn answers for the judge to read.
    q_text = "\n\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(question["turns"]))
    a_text = "\n\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(a_ans["turns"]))
    b_text = "\n\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(b_ans["turns"]))

    judge_url = "https://api.openai.com/v1" if JUDGE_PROVIDER == "openai" else JUDGE_URL
    judge_headers = (
        {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        if JUDGE_PROVIDER == "openai" else {}
    )

    resp = client.post(
        f"{judge_url}/chat/completions",
        headers=judge_headers,
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "user", "content": PAIRWISE_JUDGE_PROMPT.format(
                    question=q_text, answer_a=a_text, answer_b=b_text)}
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    rationale = resp.json()["choices"][0]["message"]["content"]

    m = re.search(r"Verdict:\s*([ABC])\b", rationale)
    if not m:
        # Judge didn't follow the format. Treat as tie — conservative.
        verdict_letter = "C"
    else:
        verdict_letter = m.group(1)

    # Map back to target/baseline accounting for swap
    if verdict_letter == "C":
        winner = "tie"
    elif verdict_letter == "A":
        winner = "baseline" if swap else "target"
    else:  # B
        winner = "target" if swap else "baseline"

    return {
        "question_id": question["id"],
        "category": question["category"],
        "winner": winner,
        "judge_rationale": rationale,
    }


def judge_all(
    questions: list[Question],
    target: list[Answer],
    baseline: list[Answer],
) -> list[Verdict]:
    """Two judge calls per question (forward + swapped order) to cancel
    positional bias. Final verdict is the majority of the two."""
    verdicts: list[Verdict] = []
    by_qid_t = {a["question_id"]: a for a in target}
    by_qid_b = {a["question_id"]: a for a in baseline}
    with httpx.Client() as client:
        for q in questions:
            print(f"  judging q{q['id']:>3} ({q['category']}) ...", flush=True)
            v1 = judge_pairwise(client, q, by_qid_t[q["id"]], by_qid_b[q["id"]], swap=False)
            v2 = judge_pairwise(client, q, by_qid_t[q["id"]], by_qid_b[q["id"]], swap=True)

            # Combine: if both agree -> use that. If disagree -> tie.
            # (Catches positional bias by treating order-dependent
            # verdicts as no-signal.)
            if v1["winner"] == v2["winner"]:
                winner = v1["winner"]
            else:
                winner = "tie"
            verdicts.append({
                "question_id": q["id"],
                "category": q["category"],
                "winner": winner,
                "judge_rationale": f"FORWARD:\n{v1['judge_rationale']}\n\nSWAPPED:\n{v2['judge_rationale']}",
            })
    return verdicts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(verdicts: list[Verdict]) -> dict[str, float]:
    """Per-category win rate for the target. Win=1, tie=0.5, loss=0."""
    by_cat: dict[str, list[float]] = collections.defaultdict(list)
    for v in verdicts:
        pts = {"target": 1.0, "tie": 0.5, "baseline": 0.0}[v["winner"]]
        by_cat[v["category"]].append(pts)
    scores = {c: sum(pts) / len(pts) for c, pts in by_cat.items()}
    scores["overall"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------

def check_regression(scores: dict[str, float], baselines: dict[str, float]) -> list[str]:
    """Return a list of failure messages. Empty = pass."""
    failures = []
    overall_baseline = baselines.get("overall", None)
    if overall_baseline is None:
        print("No baseline for 'overall' — skipping regression gate.")
        return failures
    threshold = overall_baseline * (1 - THRESHOLD_PCT / 100.0)
    if scores["overall"] < threshold:
        failures.append(
            f"REGRESSION: overall {scores['overall']:.3f} < threshold "
            f"{threshold:.3f} (baseline {overall_baseline:.3f}, "
            f"tolerance {THRESHOLD_PCT}%)"
        )
    return failures


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def upload_results(
    questions: list[Question],
    target: list[Answer],
    baseline: list[Answer],
    verdicts: list[Verdict],
    scores: dict[str, float],
) -> str:
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"))
    prefix = f"eval-results/mt-bench/{RUN_ID}"

    payload = {
        "run_id": RUN_ID,
        "target_model": TARGET_MODEL,
        "baseline_model": BASELINE_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_provider": JUDGE_PROVIDER,
        "score_mode": SCORE_MODE,
        "categories": CATEGORIES,
        "n_questions": len(questions),
        "scores": scores,
    }
    s3.put_object(
        Bucket=RESULTS_BUCKET, Key=f"{prefix}/summary.json",
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )
    # Verdicts include judge rationale — much larger, separate object.
    s3.put_object(
        Bucket=RESULTS_BUCKET, Key=f"{prefix}/verdicts.json",
        Body=json.dumps(verdicts, indent=2).encode(),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=RESULTS_BUCKET, Key=f"{prefix}/answers-target.json",
        Body=json.dumps(target, indent=2).encode(),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=RESULTS_BUCKET, Key=f"{prefix}/answers-baseline.json",
        Body=json.dumps(baseline, indent=2).encode(),
        ContentType="application/json",
    )
    return f"s3://{RESULTS_BUCKET}/{prefix}/"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"=== MT-Bench eval: {TARGET_MODEL} vs {BASELINE_MODEL} ===")
    print(f"    run_id={RUN_ID}  judge={JUDGE_PROVIDER}:{JUDGE_MODEL}")
    print(f"    score_mode={SCORE_MODE}  threshold={THRESHOLD_PCT}%\n")

    questions = load_questions()
    baselines = load_baselines()

    print("\n--- Phase 1: generating answers ---")
    target, baseline = gen_all_answers(questions)

    print("\n--- Phase 2: judging (pairwise, swap-balanced) ---")
    verdicts = judge_all(questions, target, baseline)

    print("\n--- Phase 3: scoring ---")
    scores = score(verdicts)
    for cat, s in sorted(scores.items()):
        baseline_val = baselines.get(cat)
        delta = (
            f"  (Δ vs baseline {baseline_val:.3f}: "
            f"{(s - baseline_val):+.3f})" if baseline_val is not None else ""
        )
        print(f"  {cat:14s}: {s:.3f}{delta}")

    print("\n--- Phase 4: regression gate ---")
    failures = check_regression(scores, baselines)
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("  ✓ no regression")

    print("\n--- Phase 5: upload to S3 ---")
    s3_path = upload_results(questions, target, baseline, verdicts, scores)
    print(f"  Wrote to {s3_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
