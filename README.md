# nv-graft

Run **nvidia/Qwen3.8-Flash-Next-NVFP4** — NVIDIA's official NVFP4 quant of Qwen3.8-Flash-Next — on an RTX PRO 6000 (SM120) with the [Pennyroyal SGLang fork](https://github.com/jpezzulli/sglang-rtxpro6000). It does this by grafting NVIDIA's expert tensors onto the [primitive-ai](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4) checkpoint layout that the fork loads, as a one-time offline build. No server-code changes, no fork patches.

## The problem

NVIDIA's checkpoint won't load in this stack. Its MTP drafter layer ships a per-expert tensor layout (3,072 separate expert tensors) that the fork's NEXTN speculative-decode parser doesn't know — the server fails at model load. primitive-ai's build of the same model loads perfectly but carries an older expert calibration. So the only apparent options were: serve the stale calibration, or don't serve NVIDIA's quant at all.

## Why graft

Alternatives we weighed:

- **Patch the fork's MTP loader** for the per-expert dialect — works in principle, but couples model bytes to fork maintenance; a vendor patch to carry forever.
- **Drop speculative decode** and serve NVIDIA's checkpoint without NEXTN — the per-expert MTP layer is only needed by the drafter; losing the drafter costs far more decode throughput than any calibration gain.
- **Wait for either repo to fix itself** — out of our control.

Grafting sidesteps all three: the two checkpoints' 294,912 main-model expert keys are geometry-identical (same quantization recipe), so NVIDIA's re-calibrated expert bytes are drop-in replacements inside primitive-ai's loadable skeleton — config, non-expert tensors, and its fused MTP drafter kept as-is, via symlinks. Result: NVIDIA's calibration, the fork's loadable layout, the drafter intact, zero changes to server code or launch path beyond `--model-path`. On our box: +3% decode over primitive-ai, acceptance unchanged, rollback is a file copy + restart. The graft dir is disposable — rebuildable any time NVIDIA republishes.

**Scope:** built and tested only on this exact checkpoint pair. The scripts key off their layouts (`.mlp.experts.` matching, fused-vs-per-expert asymmetry); nothing else has passed the sanity gate. Treat other models as unsupported.

## How

```bash
# both checkpoints on disk
hf download nvidia/Qwen3.8-Flash-Next-NVFP4       --local-dir /models/nvidia/Qwen3.8-Flash-Next-NVFP4 --max-workers 4
hf download primitive-ai/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 --max-workers 4

git clone https://github.com/brakthehack/nv-graft && cd nv-graft

# build (~20 min, ~65 GB free, ~10 GB peak RAM)   -> log ends: GRAFT_BUILD_DONE
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft bash scripts/run_nv_graft_build.sh

# verify                                           -> prints: VERDICT: GRAFT-SANE
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out

# serve: point your server's --model-path at OUT_DIR and restart
```

Needs a python with `safetensors` + `torch`. Stop if the build doesn't end `GRAFT_BUILD_DONE` or the gate doesn't print `GRAFT-SANE` — either way nothing has touched your server. Both sources stay read-only.

## Evidence

- **[docs/graft-report.md](docs/graft-report.md)** — full writeup: the exact launch commands (pre/post), the checkpoint-diff finding with counts, build/sanity/cutover design, pitfalls, before/after benchmarks.
- Health-gated systemd cutover with auto-rollback: [`scripts/nv-graft-cutover.sh`](scripts/nv-graft-cutover.sh) · before/after probe: [`scripts/perf_probe.py`](scripts/perf_probe.py).
- Checkpoints: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) · [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4)

*Fork versions: built and measured on the v2.3.0 (FR-Spec) line. Newer fork releases through v2.4.0 change nothing in the MTP/NEXTN load path — the per-expert dialect that forces the graft is still unsupported there — so the graft remains necessary and compatible.*
