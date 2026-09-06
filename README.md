# NV-GRAFT: grafting NVIDIA's re-calibrated NVFP4 experts onto a serving-compatible checkpoint skeleton

*A production cutover writeup. Scripts in `scripts/`, copy-paste ready. Results at the bottom.*

## TL;DR

We serve Qwen3.8-Flash-Next in NVFP4 on a single-GPU box. NVIDIA republished their NVFP4 checkpoint with **re-calibrated expert tensors** — but in a tensor dialect our serving path can't load (the MTP drafter layer changed layout). The old checkpoint we ran (primitive-ai's) loads perfectly but carries older expert calibration.

Instead of choosing between them, we **grafted**: take the primitive-ai skeleton (config, non-expert tensors, fused-MTP layer) and swap in *only* NVIDIA's expert tensors, which we verified are byte-geometry-identical where they overlap. Assembled offline as a symlink farm — ~65 GB of output, near-zero copies, both sources read-only.

Cutover landed cleanly: **decode 239 t/s** (previous checkpoint: 232; rollback bar: ≥200), speculative accept length **2.55 unchanged**, zero restarts since, one ~3.5-minute cache drop during the window.

---

## The interesting finding: two "identical" checkpoints disagree exactly where you don't look

Both repos claim to ship the same NVFP4 model. Their index key sets:

- primitive-ai: 294,914 `.mlp.experts.*` keys
- nvidia: 297,986 `.mlp.experts.*` keys

The 294,912 overlapping keys are **byte-identical in geometry** — same dtypes, same shapes, same weight/scale-block triplet structure. This is what makes a graft possible at all: the quantization recipe is the same, so NVIDIA's tensors are drop-in replacements in the skeleton's layout.

The disagreement is entirely in the **MTP drafter layer** (`mtp.layers.0.mlp.experts.*`), and it's a layout philosophy difference, not a corruption:

| | primitive-ai | nvidia |
|---|---|---|
| MTP expert layout | **fused**: `experts.gate_up_proj`, `experts.down_proj` (2 stacked tensors) | **per-expert**: `experts.0.down_proj.weight`, `experts.0.down_proj.weight_scale_inv`, … (3,072 keys) |
| Loads in our serving path (NEXTN spec-decode) | ✅ | ❌ (parser doesn't know the per-expert dialect there) |

Keeping primitive-ai's fused-MTP layer is the whole reason to graft rather than switch — the drafter is what powers speculative decode, and it's the piece we've tuned (FR-Spec token map) against.

**Lesson:** when comparing two checkpoints of "the same" model, diff the **key sets**, not just the geometry of sampled keys. Sampled-geometry equality told us the graft was viable; it did not tell us the sets were symmetric — and the first build attempt died on exactly the two keys nobody sampled. Checkpoint surgery should start with a set-difference bucketed by prefix; the symmetric difference is the whole risk surface.

## The graft, concretely

Input: `NV_DIR` (nvidia NVFP4), `PA_DIR` (primitive-ai NVFP4). Output: `OUT_DIR`, a valid HF checkpoint directory.

1. **Scope** = expert keys present in *both* indices (294,912). The 2 PA-only fused-MTP keys fall through to the skeleton by design.
2. **Pre-flight geometry gate** — sample key headers from both sources (dtype+shape); any mismatch ⇒ abort before touching heavy I/O.
3. **Extract** NV expert tensors one shard at a time (~10 GB/shard peak; a full-checkpoint load would OOM), repacking into `nv-experts-partNN.safetensors` (9 parts).
4. **Symlink farm** — every non-grafted key keeps pointing at the primitive-ai shard file. Zero-copy, sources never written. Config/tokenizer/chat-template also symlinked: the quantization dialect (NVFP4 + ignore list), BF16 PLE embedding, and BF16 MTP are the skeleton's, untouched.
5. **Rewrite** `model.safetensors.index.json` (296,474 keys) and verify every referenced shard exists → terminal token `GRAFT_BUILD_DONE`/`FAIL`.

Build: ~20 min on local NVMe. Result: 64 GB graft dir = 9 real parts + a farm of symlinks.

**Sanity gate** (`scripts/nv_graft_sanity.py`) — must pass before the graft touches a serving box:
- every index key resolves to an existing file;
- 40 sampled grafted expert keys match PA geometry (recipe preserved) *and* NV geometry (bytes are NV's);
- non-expert shards `realpath` back to PA (proves zero-copy, no silent requant);
- the 2 PA-only MTP keys resolve via symlink to PA (proves they weren't swallowed by a graft part).

Two subtleties that bit the first versions of the scripts, both instances of the same rule — *assert the asymmetry explicitly*:
- scope must be the **intersection**, not "all expert keys in the skeleton index" (else first NV lookup: `KeyError: mtp.layers.0.mlp.experts.down_proj`);
- the gate may only sample NV-comparable keys "vs NV"; the PA-kept keys need their own assertion, not the shared code path.

## Health-gated cutover with verified auto-rollback

`scripts/nv-graft-cutover.sh` changes exactly two things in the systemd unit: `--model-path` → graft dir, and strips a stale profiling flag. Everything else — 384K ctx with YaRN, hicache/NIXL tiers, NEXTN 3/1/4 spec config, mem fractions — byte-for-byte unchanged, which is what makes the before/after numbers comparable.

The gate structure is what makes a 3 a.m. unattended run acceptable:

- re-checks the build token **and** the sanity verdict before touching the unit (safe to fire by hand, defense in depth);
- backs up the unit file off-box first;
- restarts, then polls `/health_generate` for 15 min;
- on timeout: restores the backup, restarts, and **verifies the rollback itself boots healthy** before declaring anything;
- terminal states: `CUTOVER_SUCCESS` / `ROLLBACK_HEALTH_OK` / escalate.

Worst case is two bounded cache drops and the old checkpoint serving. Actual: boot ~5 min, one drop, no rollback.

## Acceptance results

Rollback trigger was a material drop in **spec accept length** — the drafter was tuned against the old expert bytes, so if re-calibration perturbed logits, accept length is where it shows first.

| metric | old checkpoint | graft | verdict |
|---|---|---|---|
| decode t/s (non-think) | 232 | **239.0** | +3% |
| decode t/s (thinking) | — | 224.2 | — |
| prefill t/s @ ~100k ctx | — | 11.2k | — |
| spec accept length / rate | 2.55 / ~0.5 | **2.55 / 0.52** | unchanged |
| NRestarts since cutover | — | 0 | clean |

Plus functional verification against the live server: model path/tokenizer resolve to the graft dir, a reasoning round-trip completes with clean `finish_reason=stop`.

**Takeaway on the model side:** the NVIDIA expert re-calibration transferred through a mixed-skeleton checkpoint without perturbing speculative acceptance at all — accept length identical, decode slightly *faster*. Different calibration, same recipe, safe drop-in. And a graft dir is cheap insurance: it's disposable, rebuildable from two read-only sources, and rollback is a file copy + restart.

## Usage

```bash
# build
NV_DIR=/models/nvidia/Qwen3.8-Flash-Next-NVFP4 \
PA_DIR=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
OUT_DIR=/models/nv-graft \
  bash scripts/run_nv_graft_build.sh

# gate
NV_DIR=... PA_DIR=... OUT_DIR=/models/nv-graft \
  python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out

# cutover (health-gated, auto-rollback)
UNIT=/etc/systemd/system/sglang-flashnext.service \
OLD_PATH=/models/primitive-ai/Qwen3.8-Flash-Next-NVFP4 \
NEW_PATH=/models/nv-graft \
HEALTH_URL=localhost:8086/health_generate \
MODELS_URL=localhost:8086/v1/models MODEL_NAME=qwen38-flashnext \
BACKUP=/models/backups/sglang.service.pre-graft \
  bash scripts/nv-graft-cutover.sh
```

Scripts are general (env-var config); they ran as-shipped against the real checkpoints on a production systemd unit. `run_nv_graft_build.sh` additionally guarantees a terminal verdict token in the log on *any* crash exit — a progress log frozen on its last line is indistinguishable from a working build, and a supervisor watching it will wait forever for a process that's been dead for an hour. That wrapper is 10 lines and earns its place.
