# Phase #83b: vllm-llama-8b → AWQ-INT4 quantization

Status: **APPLIED 2026-05-01**. S3 pre-staged via `llm/operational/awq-stage-llama-3.1-8b.yaml` (Job `awq-stage-llama-3-1-8b` ran in 5m41s, ~5.73GB AWQ weights at `s3://raj-ai-lab-eks-cilium-model-weights/llama-3.1-8b-awq/`). Code diff applied to `deployment-models.yaml` in the same commit that flipped this status line.

## What this delivers

- VRAM footprint: 16GB (FP16) → ~4-5GB (AWQ-INT4). Frees ~11GB on the L4 24GB.
- Throughput: ~2-3× tokens/sec on L4 (Marlin INT4 kernels exploit Ada's int4 tensor cores).
- KV cache pool grows commensurately: at gpu-mem-util 0.92, was ~6GB free for KV; becomes ~17GB. Translates to **3× more concurrent decode contexts per pod** (~30 → ~90 simultaneous sessions).
- Quality cost: typically <2% on standard chat benchmarks (perplexity, MMLU). For a chat-tier classifier+executor workload, indistinguishable from FP16.

## Why this is staged not committed

vLLM's `--quantization awq` flag requires the on-disk weights to be in AWQ-INT4 format. The current S3 path (`s3://raj-ai-lab-eks-cilium-model-weights/llama-3.1-8b-instruct/`) holds the FP16 safetensors. Switching the flag without staging the AWQ weights produces a load error.

## Operator pre-stage steps

```bash
# Option A: pull a community AWQ-quantized release (fastest)
# `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` is the canonical one
mkdir -p /tmp/llama-3.1-8b-awq
cd /tmp/llama-3.1-8b-awq
huggingface-cli download \
  hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir . \
  --local-dir-use-symlinks False

aws s3 sync . s3://raj-ai-lab-eks-cilium-model-weights/llama-3.1-8b-awq/

# Option B: produce our own AWQ from the existing FP16 (4-bit quantize
# pass takes ~30 min on g6.xlarge, single-shot; needs the PyTorchJob
# pattern from Phase #81c). Skipped here — option A is operationally
# cheaper.
```

## Code change to apply (gitops, after staging)

In `llm/base/deployment-models.yaml`, on the `vllm-llama-8b` Deployment:

```diff
       initContainers:
         - name: model-sync
           env:
-            - { name: S3_URI,     value: s3://raj-ai-lab-eks-cilium-model-weights/llama-3.1-8b-instruct/ }
+            - { name: S3_URI,     value: s3://raj-ai-lab-eks-cilium-model-weights/llama-3.1-8b-awq/ }
             - { name: MODEL_PATH, value: /model }
       containers:
         - name: vllm
           args:
             - "--model"
             - "/model"
             - "--served-model-name"
             - "llama-3.1-8b"
             - "--tensor-parallel-size"
             - "1"
-            - "--dtype"
-            - "half"
+            - "--quantization"
+            - "awq_marlin"   # NOT plain "awq" — see kernel-path note below
+            - "--dtype"
+            - "half"  # FP16 activations on top of INT4 weights — the
+                      # canonical AWQ inference dtype
             ...
           resources:
-            requests: { nvidia.com/gpu: "1", cpu: "2", memory: "8Gi" }
-            limits:   { nvidia.com/gpu: "1", cpu: "3", memory: "28Gi" }
+            # Smaller — AWQ load uses ~5GB peak host RAM (vs ~10-12GB
+            # for FP16). KV cache still benefits from full GPU memory.
+            requests: { nvidia.com/gpu: "1", cpu: "2", memory: "5Gi" }
+            limits:   { nvidia.com/gpu: "1", cpu: "3", memory: "16Gi" }
```

## vLLM AWQ kernel-path footgun

vLLM 0.7.2 has TWO AWQ code paths and the choice is load-bearing:

- `--quantization awq` — legacy reference implementation. vLLM
  warns on startup: *"awq quantization is not fully optimized
  yet. The speed can be slower than non-quantized models."*
  This is the path the original Phase #83b commit (f5ea410)
  accidentally selected; the live pod was potentially SLOWER
  than the FP16 baseline it replaced.
- `--quantization awq_marlin` — Marlin kernel path. INT4 weights
  × FP16 activations on Ada/Hopper int4 tensor cores. This is
  what delivers the advertised ~2-3× throughput.

vLLM's startup log makes the right answer explicit:
```
awq_marlin.py:115] Detected that the model can run with
awq_marlin, however you specified quantization=awq explicitly,
so forcing awq. Use quantization=awq_marlin for faster inference.
```

Always use `awq_marlin`. Verified by greppinng pod logs:
`grep -E "Marlin|awq_marlin" pod-logs.txt` should match a
"Loading model weights took ..." that's followed by Marlin-kernel
init lines, not the legacy awq config-warning.

## Validation post-apply

```bash
# 1. Pod cold-starts cleanly (vLLM's startup logs should mention
#    "Quantization=AWQ" + "AWQ kernel selected: marlin")
kubectl -n llm logs -l app=vllm-llama-8b -c vllm | grep -iE "quant|awq|marlin"

# 2. Concurrent throughput: send N requests, observe vllm_num_requests_running
#    metric. Should show 30+ concurrent vs ~10 with FP16.
kubectl -n monitoring exec sts/prometheus-kube-prometheus-stack-prometheus -c prometheus \
  -- wget -qO- 'http://localhost:9090/api/v1/query?query=vllm:num_requests_running{pod=~"vllm-llama-8b-.*"}'

# 3. Quality regression check: run the existing RAGAS workflow (eval/runs/
#    ragas-eval.yaml). Faithfulness/answer-relevancy should be within
#    5pp of the FP16 baseline (eval/ragas/baselines.json).
```

## Cost-share update post-Phase-#83b

vllm-llama-8b currently:
- 1 always-warm pod (KEDA cron trigger)
- ~10-15 concurrent generations per pod
- Handles ~70% of trivial-tier traffic at 1000 users (~50 concurrent)
- → ~3-4 pods needed at peak (Phase #80c HPA scales 1→4)

With AWQ:
- 1 always-warm pod
- ~30-90 concurrent generations per pod
- 1 pod handles peak trivial-tier load
- HPA still scales to 4 if real burst, but rarely fires

**Effective savings**: scale-up events become rare → fewer Karpenter g6.xlarge spawns → ~$50/mo less in scale-up GPU spend at 1000-user load. Combined with Phase #80d's off-hours scale-to-0, the AWQ cost story is "1 pod 24/5, near-zero scale-up under typical load."
