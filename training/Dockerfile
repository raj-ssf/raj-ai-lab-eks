# Training image — Axolotl-based fine-tuning runner.
#
# Base: axolotlai/axolotl pinned to a known-good tag. Provides PyTorch +
# transformers + peft + bitsandbytes + flash-attn + the `axolotl` CLI.
# Pulled by the GHA build, not by Kubernetes — so it does not need to be
# on the Kyverno catchall allowlist (only the resulting ECR-pushed image
# does, and that's already covered via aws_ecr_repository.training).
#
# What we add on top:
#   - awscli + boto3 + s3fs: needed for `aws s3 sync` of trained adapters
#     back to the training bucket, plus boto3 for any in-process S3 I/O.
#   - configs/: the Axolotl YAML lives in /workspace/configs/, referenced
#     from the PyTorchJob command.
#
# What we do NOT bake in:
#   - Datasets: pulled at training time from HF Hub (tatsu-lab/alpaca is
#     public, no auth). For private datasets we'd switch to s3:// URIs and
#     mount via Pod Identity creds.
#   - Model weights: pulled at training time. Llama 3.1 8B comes from
#     NousResearch/Meta-Llama-3.1-8B (open mirror — no HF token needed).
#     ~16GB pull on cold start; cached on the node after first run if
#     Karpenter doesn't recycle the node.

FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

# Install AWS tooling. `awscli` is the v1 pip package — fine for `aws s3
# sync`. `boto3` is for any future in-process S3 I/O. `s3fs` lets us
# pass s3:// paths directly into HF datasets if we move datasets to S3.
RUN pip install --no-cache-dir \
        awscli \
        boto3 \
        s3fs

# Workdir aligns with where Axolotl expects to find configs + write
# outputs. Job manifest mounts nothing — output stays in the container's
# filesystem until `aws s3 sync` ships it to the bucket.
WORKDIR /workspace

# Configs only — runs/ stays in git, not in the image. The PyTorchJob
# manifest lives in raj-ai-lab-eks/training/runs/ for kubectl apply.
COPY configs/ /workspace/configs/

# No CMD/ENTRYPOINT — the PyTorchJob spec sets the command. We don't
# want a default that could accidentally run training on `docker run`.
