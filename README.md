# nv-graft

Get NVIDIA's re-calibrated NVFP4 experts into a checkpoint your serving stack can actually load — without giving up the drafter that powers your speculative decode.

## Why

You serve an NVFP4 model that loads fine from a community checkpoint (primitive-ai), but NVIDIA has republished it with better expert calibration — in a tensor layout your loader can't parse. Grafting gives you both: NVIDIA's expert bytes inside the loadable skeleton, including its fused MTP drafter. ~64 GB output, near-zero copies (symlink farm), both sources stay read-only, rollback is a file copy + restart. On our box: +3% decode, speculative acceptance unchanged.

**Scope: tested only on Qwen3.8-Flash-Next-NVFP4** (nvidia's and primitive-ai's builds of it). The scripts key off that checkpoint pair's layout — nothing here has been run against any other model. The *pattern* (graft re-calibrated experts onto a loadable skeleton) should generalize to any two same-recipe quantized checkpoints, but consider that a hypothesis, not support: expect to adapt the graft scope (`.mlp.experts.` matching, fused-vs-per-expert asymmetry handling) and re-verify with the sanity gate before serving anything else.

## How

```bash
# both checkpoints on disk
hf download nvidia/Qwen3.8-Flash-Next-NVFP4       --local-dir /models/nvidia/Qwen3.8-Flash-Next-NVFP4 --max-workers 4
hf download primitive-ai/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 --max-workers 4

git clone https://github.com/brakthehack/nv-graft && cd nv-graft

# build (~20 min)        -> log ends: GRAFT_BUILD_DONE
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft bash scripts/run_nv_graft_build.sh

# verify                 -> prints: VERDICT: GRAFT-SANE
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out

# serve: point your server's --model-path at OUT_DIR and restart
```

Stop if the build doesn't end `GRAFT_BUILD_DONE` or the gate doesn't print `GRAFT-SANE` — either way nothing has touched your server. Needs a python with `safetensors` + `torch`, ~65 GB free, ~10 GB peak RAM.

## Evidence & references

- **[docs/graft-report.md](docs/graft-report.md)** — full writeup: pre/post launch commands, the checkpoint-diff finding, build/sanity/cutover design, pitfalls, benchmark tables.
- Health-gated systemd cutover with auto-rollback: [`scripts/nv-graft-cutover.sh`](scripts/nv-graft-cutover.sh). Before/after probe: [`scripts/perf_probe.py`](scripts/perf_probe.py).
- Serving fork: [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) (Pennyroyal). Stock SGLang also works — the graft is just a checkpoint dir; drop the fork-only flags noted in the report.
- Source checkpoints: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) · [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4)
