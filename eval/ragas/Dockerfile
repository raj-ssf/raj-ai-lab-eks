# RAGAS eval image — RAG pipeline regression gate.
#
# What this is for:
#   Phase #3 of the post-Phase-T plan: wire RAGAS into CI so every
#   rag-service / langgraph-service / dataset PR gets graded for
#   faithfulness, answer_relevancy, and context_precision against a
#   checked-in golden dataset (eval/ragas/dataset.yaml). Score
#   regressions >5pp fail the workflow.
#
# Why a separate image (not extending eval/Dockerfile):
#   The F4 lm-eval image pins lm_eval==0.4.7 + transformers>=4.45 + a
#   particular langfuse v2 SDK. RAGAS pulls langchain 0.3.x + pydantic
#   2.9 + datasets 3.x. The two stacks are pip-resolvable in isolation
#   but together they introduce conflicts (langchain pulls a newer
#   pydantic than lm_eval's old pin allows). Cleaner to keep them in
#   separate images with separate ECR repos and separate build
#   workflows.
#
# Runtime shape:
#   The eval Job mints a Keycloak token via the test-user direct-access
#   grant (read from a K8s Secret), POSTs each dataset question to
#   langgraph-service /invoke (preserving session_id so RAG retrieval
#   fires), captures the response + retrieved_chunks, and feeds both
#   into RAGAS's metric set. The judge LLM is the same in-cluster
#   vllm-llama-8b that the trivial tier already uses; embeddings reuse
#   vllm-bge-m3.

FROM python:3.11-slim

# RAGAS itself doesn't need the AWS CLI — boto3 is sufficient for the
# single PUTs we do at the end. So no CLI install step (saves ~150 MB
# vs the lm-eval image).

# Build deps for some of langchain's transitive native packages
# (e.g. orjson, pydantic_core wheels). Slim image misses them.
# `--no-install-recommends` keeps the layer thin.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps in a separate layer from the source so source-
# only edits don't bust the pip cache.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Phase #46e (2026-05-03): pre-cache the HHEM model into the image so
# the eval Job runs offline. RAGAS's FaithfulnesswithHHEM loads
# vectara/hallucination_evaluation_model on first use; we don't want
# that download in the hot path (HF can be slow / rate-limited; we
# also want determinism on the model version baked into a given
# image SHA).
#
# trust_remote_code=True is required because HHEM v2.x ships custom
# modeling code in the repo. Set HF_HOME so the cache lives somewhere
# stable across runtime invocations; HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
# below force the loader to use only what's in the cache, never reach
# the network.
ENV HF_HOME=/opt/hf-cache
RUN mkdir -p "$HF_HOME" && \
    python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
m = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True); \
t = AutoTokenizer.from_pretrained('vectara/hallucination_evaluation_model'); \
print('HHEM cached:', m.__class__.__name__, t.__class__.__name__)"

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# After requirements settle, drop the build-essential layer to slim the
# final image. Most Python wheels are pre-built; build-essential is
# only needed when a sdist falls through. If pip resolves cleanly to
# wheels, this purge keeps the runtime image small.
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Eval source — runner + dataset + baselines. Keeping them in the
# image (rather than mounting from a ConfigMap) means the image SHA is
# the deterministic record of "what dataset version was scored" — a
# rebuilt image with edited dataset.yaml gets a fresh tag.
COPY run_ragas.py    /workspace/run_ragas.py
COPY dataset.yaml    /workspace/dataset.yaml
COPY baselines.json  /workspace/baselines.json

# Job spec sets the command. Skipping CMD avoids surprise execution on
# a bare `docker run` for local debugging.
