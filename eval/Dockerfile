# Eval image — lm-evaluation-harness + Langfuse SDK + AWS tools.
#
# What this is for:
#   F4 of fine-tuning: compare the F2-trained LoRA adapter to the base
#   model on standard benchmarks (MMLU, ARC, HellaSwag, etc.). The
#   eval Job hits the in-cluster vllm-llama-8b Service over HTTP using
#   lm-eval's `local-completions` model adapter (vLLM exposes both
#   `llama-3.1-8b` and `llama-3.1-8b-alpaca` after F3).
#
# Why slim Python base + lm-eval (not vLLM/torch):
#   The eval workload is CPU-bound on the eval-side — it's just an
#   HTTP client generating prompts and grading responses. All the GPU
#   work happens in the vLLM pod we're calling. So the eval container
#   doesn't need CUDA or PyTorch — keeps the image ~500MB instead of
#   the multi-GB ML images. Faster build, faster pull.

FROM python:3.11-slim

# AWS CLI v2 install (Debian package isn't shipped, use the official
# bundle). Smaller alternative than awscli pip package which is v1.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip ca-certificates && \
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    apt-get purge -y unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Python deps:
#   - lm-eval[api]: lm-evaluation-harness + the local-completions / openai
#     model adapters (HTTP backends, no torch/vllm dep)
#   - transformers>=4.45.0 explicit pin: lm-eval 0.4.7's hf_vlms.py
#     module references transformers.AutoModelForVision2Seq at class-
#     body load time, and that attribute was added to transformers in
#     4.45 (Oct 2024). Without this pin, pip resolves an older
#     transformers and every `lm_eval` invocation crashes on import
#     with `AttributeError: module transformers has no attribute
#     AutoModelForVision2Seq`. Affects API-only usage too because
#     lm_eval/__init__.py forces the model-adapter package imports.
#   - langfuse: SDK for emitting trace summaries (optional at runtime —
#     the wrapper script no-ops if LANGFUSE_PUBLIC_KEY env isn't set)
#   - boto3: programmatic S3 if the script needs it (awscli handles the
#     bulk sync at the end)
RUN pip install --no-cache-dir \
        "lm-eval[api]==0.4.7" \
        "transformers>=4.45.0,<5.0" \
        langfuse==2.60.0 \
        boto3

WORKDIR /workspace

# Wrapper script that runs both evals (base + LoRA) in parallel against
# the same vLLM Service, parses results, optionally pushes a Langfuse
# trace, and syncs everything to S3.
COPY scripts/run-eval.py /workspace/run-eval.py

# No CMD — the Job spec sets the command. Avoids accidental runs on
# `docker run`.
