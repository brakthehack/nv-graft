# NV-GRAFT: grafting NVIDIA's re-calibrated NVFP4 experts onto a serving-compatible checkpoint skeleton

*A production cutover writeup — the evidence page. For the how-to, see the [main README](../README.md).*

**Environment this was productionized on**

- Serving fork: [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) ("Pennyroyal", v2.1.1 lineage + local patches; our production tree runs v2.3.0's FR-Spec release). Stock SGLang works for the graft itself — drop the fork-only flags listed below.
- Source checkpoints: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) · [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4)
- FR-Spec draft token map (the +25% decode step, separate from the graft): in the fork at `configs/pennyroyal/frspec/flash-next-64k.pt`
- NIXL cache-tier backend config (fork-only): `configs/pennyroyal/nixl-posix.toml`
- Live unit this ran on: `/etc/systemd/system/sglang-flashnext.service` (systemd, `Restart=on-failure`, HF offline mode, local compiler/HF cache dirs)

**Support scope:** everything here is tested against exactly one checkpoint pair — nvidia's and primitive-ai's Qwen3.8-Flash-Next-NVFP4. Treat other models as unsupported until the sanity gate says otherwise; see the Scope note in the [main README](../README.md).

## TL;DR

We serve Qwen3.8-Flash-Next in NVFP4 on a single-GPU box. NVIDIA republished their NVFP4 checkpoint with **re-calibrated expert tensors** — but in a tensor dialect our serving path can't load (the MTP drafter layer changed layout). The old checkpoint we ran (primitive-ai's) loads perfectly but carries older expert calibration.

Instead of choosing, we **grafted**: primitive-ai skeleton (config, non-expert tensors, fused-MTP drafter) + NVIDIA's expert tensors, which we verified are byte-geometry-identical where they overlap. Assembled offline as a symlink farm — ~64 GB output, near-zero copies, both sources read-only.

Cutover landed cleanly. Headline: **decode 232 → 239 t/s** under the same spec-decode config, accept length unchanged (probe snapshot 2.55 vs 2.55; live traffic trending 3.03), zero restarts, one ~3.5-minute cache drop during the window. Full benchmark tables at the bottom.

---

## Quickstart

**Prereqs:** both checkpoints on disk (fetch with `hf download nvidia/Qwen3.8-Flash-Next-NVFP4` and `hf download primitive-ai/Qwen3.8-Flash-Next-NVFP4`), a Python with `safetensors`+`torch`, ~65 GB free, ~10 GB peak RAM. Sources are never written.

```bash
git clone https://github.com/brakthehack/nv-graft && cd nv-graft

# 1. build the graft dir (~20 min)         -> last log line: GRAFT_BUILD_DONE
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft bash scripts/run_nv_graft_build.sh

# 2. sanity gate                            -> prints: VERDICT: GRAFT-SANE
#    (redirect matters: step 3's cutover script gates on this file existing)
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out

# 3. point your server at OUT_DIR (edit unit's --model-path, restart, wait for /health_generate 200)
#    or run scripts/nv-graft-cutover.sh for the health-gated auto-rollback version — see "Reproduce the graft"

# 4. benchmark                             -> decode/prefill JSON lines, see Benchmarks
BASE_URL=http://127.0.0.1:8086 MODEL=qwen38-flashnext TOKDIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
python3 scripts/perf_probe.py
```

Stop if step 1 doesn't end `GRAFT_BUILD_DONE` or step 2 doesn't print `GRAFT-SANE` — either failure means the graft dir is not loadable and nothing has touched your server.

---

## Hardware & software spec (the box these numbers came from)

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB VRAM), driver 610.43.03, `tp=1`
- CPU: AMD EPYC 7663 (56c/112t), 251 GB RAM
- Storage: local NVMe (model + hierarchical-cache tiers)
- Serving: SGLang-based fork (v2.3.0 lineage + local patches), Python venv with `safetensors` + `torch`
- Model dialect: NVFP4 (`--quantization modelopt_fp4`), BF16 KV, BF16 PLE embedding, BF16 fused MTP drafter
- Post-graft KV pool: `max_total_num_tokens = 428,352` @ 384K ctx, `page-size 64`

## The base version (pre-graft production)

Two checkpoints, one serving path:

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir /models/nvidia/Qwen3.8-Flash-Next-NVFP4 --max-workers 4
# primitive-ai/Qwen3.8-Flash-Next-NVFP4 previously fetched the same way
```

Launch command (systemd `ExecStart`, one line; box-specific paths are `<>`-marked, everything else verbatim):

```bash
python -m sglang.launch_server \
  --model-path <primitive-ai checkpoint dir> \
  --load-format safetensors --served-model-name qwen38-flashnext \
  --host 0.0.0.0 --port 8086 \
  --tp 1 --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype bf16 \
  --mem-fraction-static 0.981 --context-length 384000 \
  --json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
  --page-size 64 --max-running-requests 4 --sleep-on-idle --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 24 \
  --gdn-mtp-cache-mode none --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval 64 \
  --enable-hierarchical-cache --hicache-size 32 --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first \
  --hicache-storage-backend nixl --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config @<nixl-posix backend toml> \
  --ple-offload-embedding --trust-remote-code \
  --chat-template <checkpoint dir>/chat_template.jinja \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \
  --speculative-token-map <fr-spec token map .pt> \
  --expert-distribution-recorder-mode stat \
  --watchdog-timeout 1800
```

Notes on reproducibility:
- The hierarchical-cache/NIXL tier, `--gdn-mtp-cache-mode`, `--ple-offload-embedding`, and the `--speculative-token-map` are features of our fork — on stock SGLang, drop those flags; the graft itself is fork-agnostic (it's just a checkpoint directory).
- YaRN override (262144 → 384000 ctx, factor 2.0) is the production context recipe; without it the server boots at native 262K.
- The final cutover additionally **stripped `--expert-distribution-recorder-mode stat`** (leftover MoE profiling instrumentation — see the delta table below).

## The final command (post-graft production)

The version the box actually runs after cutover — identical to the base command except `--model-path` now points at the graft dir (`OUT_DIR`) and `--expert-distribution-recorder-mode stat` is gone. Verified verbatim against the live unit:

```bash
python -m sglang.launch_server \
  --model-path <graft dir (OUT_DIR)> \
  --load-format safetensors --served-model-name qwen38-flashnext \
  --host 0.0.0.0 --port 8086 \
  --tp 1 --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype bf16 \
  --mem-fraction-static 0.981 --context-length 384000 \
  --json-model-override-args '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
  --page-size 64 --max-running-requests 4 --sleep-on-idle --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 24 \
  --gdn-mtp-cache-mode none --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer \
  --mamba-track-interval 64 \
  --enable-hierarchical-cache --hicache-size 32 --hicache-host-memory-mode cache \
  --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first \
  --hicache-storage-backend nixl --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend-extra-config @<nixl-posix backend toml> \
  --ple-offload-embedding --trust-remote-code \
  --chat-template <graft dir>/chat_template.jinja \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --speculative-draft-model-quantization unquant \
  --speculative-token-map <fr-spec token map .pt> \
  --watchdog-timeout 1800
```

(Our unit additionally pins `HF_HUB_OFFLINE=1` and redirects all compiler/HF caches to a local dir — operational hygiene, not part of the graft. The live `--chat-template` still names the primitive-ai path directly; the graft dir's `chat_template.jinja` is a symlink to the same bytes, so either path is equivalent.)

## The interesting finding: two "identical" checkpoints disagree exactly where you don't look

Both repos claim to ship the same NVFP4 model. Their index key sets:

- primitive-ai: 294,914 `.mlp.experts.*` keys
- nvidia: 297,984 `.mlp.experts.*` keys

The 294,912 overlapping keys are **byte-identical in geometry** — same dtypes, same shapes, same weight/scale-block triplet structure (a later audit diffed headers for **all** 294,912 overlap keys, not just samples: 0 mismatches). This is what makes a graft possible: same quantization recipe, so NV's tensors are drop-in replacements in the skeleton's layout.

The expert-set disagreement is the **MTP drafter layer** (`mtp.layers.0.mlp.experts.*`) — a layout philosophy difference, not corruption:

| | primitive-ai | nvidia |
|---|---|---|
| MTP expert layout | **fused**: `experts.gate_up_proj`, `experts.down_proj` (2 stacked tensors) | **per-expert**: `experts.0.down_proj.weight`, `experts.0.down_proj.weight_scale_inv`, … (3,072 keys) |
| Loads in our serving path (NEXTN spec-decode) | ✅ | ❌ (parser doesn't know the per-expert dialect there) |

Keeping primitive-ai's fused MTP is the whole reason to graft instead of switching — the drafter powers speculative decode and we tuned our token map against it.

(Honest footnote from a later full-set audit: "entirely" was too strong for the *whole-index* diff. Beyond the MTP expert keys, NV's index carries one extra key — `…ple_embedding.ngram_embedding.weight_scale` — and the layer-1 PLE ngram tables genuinely differ between the two (NV ships FP8 + a separate scale tensor; PA ships BF16 pre-scaled, hence PA 8.0 GB vs NV 4.0 GB per shard). The graft keeps the PA PLE dialect wholesale, which is also what our `--ple-offload-embedding` path expects. The expert-key story above stands unchanged.)

**Lesson:** when comparing two checkpoints of "the same" model, diff the **key sets**, not just the geometry of sampled keys. Sampled-geometry equality proved the graft viable; it did *not* prove the sets symmetric — the first build attempt died on exactly the two keys nobody sampled (`KeyError: mtp.layers.0.mlp.experts.down_proj`). Start checkpoint surgery with a set-difference bucketed by prefix; the symmetric difference is the whole risk surface.

## What changed, base → final (complete delta)

1. **Model directory**: `primitive-ai/Qwen3.8-Flash-Next-NVFP4` → `nv-graft/` (built by `scripts/build_nv_graft.py`). In the graft: config.json, tokenizer, chat template, PLE/attention/linear-attn/hyper-connection tensors, and the fused BF16 MTP layer = **symlinks to primitive-ai (byte-identical)**; all 294,912 main-model expert tensors = **NVIDIA's re-calibrated bytes**; index rewritten (296,474 keys).
2. **Launch flags**: removed `--expert-distribution-recorder-mode stat`. Nothing else touched — context, YaRN, cache tiers, NEXTN 3/1/4, mem fractions all identical. (This flag strip shipped in the same window as an approved cleanup; treat it as a separate, optional step — our post-hoc read is that its effect on throughput was minor-to-none; the checkpoint swap is the real delta.)
3. **Everything else on the box**: unchanged. Same unit file (backed up for one-copy rollback), same venv, same served model name → zero client changes.

## Reproduce the graft

```bash
# 1. build (~20 min on local NVMe, ~65 GB free, ~10 GB peak RAM)
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 \
PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft \
  bash scripts/run_nv_graft_build.sh          # ends GRAFT_BUILD_DONE | GRAFT_BUILD_FAIL

# 2. sanity gate (must print VERDICT: GRAFT-SANE)
NV_DIR=... PA_DIR=... OUT_DIR=/models/nv-graft \
  python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out

# 3. cutover (health-gated, auto-rollback) — or just edit your unit by hand:
#    --model-path -> /models/nv-graft, restart, poll /health_generate
UNIT=/etc/systemd/system/sglang-flashnext.service \
OLD_PATH=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
NEW_PATH=/models/nv-graft \
HEALTH_URL=localhost:8086/health_generate \
MODELS_URL=localhost:8086/v1/models MODEL_NAME=qwen38-flashnext \
BACKUP=/models/backups/sglang.service.pre-graft \
  bash scripts/nv-graft-cutover.sh
```

Build design (all in `scripts/build_nv_graft.py`, commented): graft scope = expert keys present in **both** indices; pre-flight geometry gate before heavy I/O; one NV shard loaded at a time (memory-safe); symlink farm for everything not grafted (zero-copy, sources read-only); terminal verdict token in the polled log.

Sanity gate asserts: every index key resolves; 40 sampled grafted experts match PA geometry (recipe preserved) *and* NV geometry (bytes are NV's); non-expert shards `realpath` back to PA; the 2 PA-only fused-MTP keys symlink to PA (not swallowed by a graft part).

## Benchmark methodology

`scripts/perf_probe.py` (stdlib + transformers only): cold prefill at ~6K/50K/100K tokens (random-salted prompts so radix/hierarchical caches can't fake warm numbers), decode at 512 tokens × thinking/non-thinking arms, occupancy checked before measuring. Spec-decode stats from `/metrics` and 45k-batch journal averages. **All three runs below used the identical probe, same box, same day, server otherwise quiescent.**

## Benchmarks — before and after

| run | checkpoint | spec token-map | fr-spec | decode non-think | decode think | prefill @6K/50K/100K |
|---|---|---|---|---|---|---|
| **baseline** | primitive-ai | default | ❌ | 184.3 | 179.3 | 9993 / 10685 / 10910 |
| **pre-graft prod** | primitive-ai | FR-Spec map | ✅ | 231.7 | 228.2 | 10549 / 11085 / 11156 |
| **post-graft (final)** | **nv-graft** | FR-Spec map | ✅ | **239.0** | **224.2** | 8572 / 11477 / 11204 |

(Prefill caveat, stated honestly: single-shot cold arms carry real jitter at short context — the post-graft 6K reading (8572) is *below* pre-graft (10549) while 50K/100K read slightly above. With reps=1 and one cold miss each, we treat sub-10K prefill as noise-equivalent; the 100K arm is the stable long-context comparison and it's flat (11156 → 11204). The graft changes expert bytes only; there's no mechanism for it to move prefill by double digits either way, which is itself a sanity signal.)

Accept-length (speculative decoding quality — the rollback trigger):

| window | mean accept_len | evidence |
|---|---|---|
| pre-graft, primitive-ai + FR-Spec | 2.70 | journal avg, 45,441 decode batches |
| post-graft, acceptance-probe snapshot | 2.55 | /metrics right after cutover probes |
| post-graft, live mixed traffic (next ~30 min) | **3.03** (rate 0.68) | /metrics cumulative |

Context for the tables: the baseline→pre-graft step is a *different* optimization (our FR-Spec draft token map, +25% decode) — shown because both checkpoint deltas were measured against that production line. The graft step itself: **+3% decode non-think** (231.7→239.0), think arm roughly flat, cold-prefill mid/long-context slightly up, TTFT unchanged (~0.2 s warm). Acceptance is statistically unchanged — different traffic mix moves the point estimate more than the checkpoint swap does. NRestarts 0 since cutover; health 200 throughout the poll; boot took ~5 min (the one cache drop).

**Model-side takeaway:** the NVIDIA expert re-calibration transferred through a mixed skeleton with no perturbation to speculative acceptance and a small decode gain. Different calibration, same recipe, safe drop-in — and the graft is disposable: rebuildable from two read-only sources, rollback is a file copy + restart.

## Script notes / pitfalls

- **Scope must be the intersection** of the two indices — "all expert keys in the skeleton" hits a KeyError on the first NV lookup (fused-MTP asymmetry).
- **The gate samples only NV-comparable keys** for the vs-NV check; PA-kept keys get their own realpath assertion. A gate is pipeline code — crash-test it like the thing it gates.
- **Terminal token on every exit path.** `run_nv_graft_build.sh` guarantees the polled log ends `GRAFT_BUILD_DONE/FAIL` even when Python dies on stderr; a progress log frozen on its last line is indistinguishable from a live build, and a supervisor watching it waits forever for a dead process. Ten lines, earns its place in any unattended pipeline.
- **Probe pitfalls baked into `perf_probe.py`:** local tokenization (some forks' `/encode` 400s), random-salted cold prompts (caches fake warm numbers), count `content+reasoning_content` with `ignore_eos` + `include_usage`, and check `num_running_reqs` before measuring through someone else's generation.
