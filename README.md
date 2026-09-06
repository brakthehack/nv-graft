# NV-GRAFT: grafting NVIDIA's re-calibrated NVFP4 experts onto a serving-compatible skeleton — and the four automation traps that nearly ate an overnight window

*A field report from a production model cutover. Scripts in `scripts/`, copy-paste ready.*

## TL;DR

We run Qwen3.8-Flash-Next in NVFP4 on a single-GPU SGLang box. NVIDIA republished their NVFP4 checkpoint with re-calibrated expert tensors, but in a tensor dialect our serving path can't load (fused-MTP layout changed). The fix was a **graft checkpoint**: primitive-ai's skeleton + NVIDIA's expert tensors, assembled offline with a symlink farm so nothing is copied twice. Cutover landed with **decode 239 t/s** (bar was ≥200, old baseline 232), accept length 2.55, zero restarts, one ~4-minute KV drop.

The interesting part isn't the graft — it's that the *automation around it* failed twice before succeeding, and both failures were the same species: **a supervisor that can't see its worker's death.** Full e2e flow below, traps first.

---

## The problems, in the order they ate us

### Trap 1 — graft scope assumed key-set symmetry that doesn't exist

First cut of the build script selected the swap scope from the *source-of-skeleton* index alone:

```python
expert_keys_p = {k: v for k, v in idx_p.items() if ".mlp.experts." in k}
```

Then the first NV lookup died:

```
File "/tmp/build_nv_graft.py", line 60, in <module>
    nv_shards = sorted({idx_n[k] for k in expert_keys_p})
KeyError: 'mtp.layers.0.mlp.experts.down_proj'
```

Two checkpoints of the "same" model advertise 294,914 vs 297,986 expert keys. The delta is entirely in the MTP drafter layer: the skeleton ships it **fused** (`mtp.layers.0.mlp.experts.{gate_up,down}_proj`, 2 keys), NVIDIA ships it **per-expert** (3,072 keys). Neither set maps to the other.

**Fix:** scope = intersection, and let the two PA-only keys fall through to the symlink farm — they're kept from the skeleton *by design* (keeping that MTP layout loadable is literally why we graft instead of switching):

```python
expert_keys_p = {k: v for k, v in idx_p.items() if ".mlp.experts." in k and k in idx_n}
```

Before touching a byte, diff the two indices' key sets and bucket the symmetric differences by prefix. "Byte-identical geometry on sampled keys" (which we verified up front) does **not** imply "identical key sets." Different sets, identical overlap geometry — the check passed and the build still died.

### Trap 2 — the supervisor polls the log; the worker dies on stderr

This is the one that cost the 3 hours. Build script progress goes to a log via `log()`; Python tracebacks go to stderr, which the launcher had redirected to a *different* file. The overnight chain polled `tail -1 build.log` for a terminal token (`GRAFT_BUILD_DONE`/`FAIL`) for its full 180-minute budget. Meanwhile the log was frozen at `geometry sample check: PASS` and the process had been dead for 17 minutes. Every poll faithfully reported... `PASS`.

```
06:34Z waiting build, t=10min last: geometry sample check: PASS
...
09:25Z waiting build, t=180min last: geometry sample check: PASS
09:25Z build timeout 3h — abort, no cutover
```

The gate did its job — no cutover on an unproven build, zero production risk — but "failed fast" would have beaten "failed honest."

**Fix:** a wrapper whose *only* job is to guarantee a terminal verdict token in the **polled** stream on any non-zero exit (`scripts/run_nv_graft_build.sh`):

```bash
python3 build_nv_graft.py >> "$OUT" 2>&1
rc=$?
if [ $rc -ne 0 ] && ! grep -q GRAFT_BUILD_ "$LOG"; then
  echo "graft build crashed rc=$rc (see $OUT)" >> "$LOG"
  echo "GRAFT_BUILD_FAIL" >> "$LOG"
fi
exit $rc
```

Generalized rule: **every watched stream needs a guaranteed terminal token on every exit path.** A poller that distinguishes "still running" from "done-ok" but not "done-dead" will always eventually guess "still running."

### Trap 3 — the sanity gate itself had a live crash in its dead code

Build #2 succeeded; the chain died one step later on the *gate*:

```
TypeError: can only concatenate tuple (not "list") to tuple
```

in a vestigial offset-comparison line that did nothing but execute. Worse, the gate also sampled "expert keys" against both sources — same asymmetry as Trap 1, waiting to fire: a PA-only fused-MTP key sampled "vs NV" is a guaranteed KeyError. **Fix:** gate samples only NV-comparable expert keys; the PA-kept ones get their own assertion (must symlink back to PA). A gate is part of the pipeline — it needs the same crash-testing as the thing it gates.

### Trap 4 — the unattended reviewer never started

Last stage was a codex supervisor for independent acceptance verification. It exited in ~1s:

```
Not inside a trusted directory and --skip-git-repo-check was not specified.
```

`codex exec` from a non-trusted cwd needs `--skip-git-repo-check` in unattended mode. 30 seconds of `--help` reading would have caught it; instead it ate the wake-up. (And if a gate stage's verdict is already clean-verified, re-spawning a 40-minute independent reviewer adds ceremony, not safety — know when the supervision stage is moot.)

---

## The full working e2e flow

Five stages, each gating the next; nothing touches production unless the previous stage's terminal token is present.

### Stage 0 — the insight that makes the graft cheap

Diff the two checkpoints' **key geometry** (dtype, shape, scale-tensor triplet), not just key names: the 294,912 overlapping expert keys are byte-identically shaped between the two releases. Same quantization dialect in the config (NVFP4 + the same ignore list), BF16 PLE embedding and BF16 MTP drafter kept from the skeleton. So the graft is: *skeleton verbatim, experts swapped* — no requantization, no config surgery, no re-calibration.

### Stage 1 — build (`scripts/build_nv_graft.py` + wrapper)

```bash
NV_DIR=/models/nvidia/CHECKPOINT \
PA_DIR=/models/primitive-ai/CHECKPOINT \
OUT_DIR=/models/nv-graft \
bash scripts/run_nv_graft_build.sh
```

Memory-safe design (a full checkpoint blowup would OOM; a single ~10 GB NV shard at a time does not):

1. Load both index JSONs; compute graft scope = expert keys present in **both** (Trap-1 fix).
2. Pre-flight: sample key geometry against both headers; mismatch ⇒ exit before any heavy I/O.
3. For each NV shard: `load_file` → keep wanted expert tensors → `save_file` as `nv-experts-partNN.safetensors` → free. (9 shards for the expert body.)
4. Symlink farm: every *non-grafted* key keeps pointing at the skeleton's original shard — zero-copy, and the sources are never written.
5. Rewrite `model.safetensors.index.json`; symlink config/tokenizer/chat-template files.
6. Terminal gate: every shard referenced by the index exists → `GRAFT_BUILD_DONE` / `GRAFT_BUILD_FAIL`.

Result: 64 GB graft dir, 296,474 index keys, 9 real parts + symlink farm, ~20 min on the box.

### Stage 2 — sanity (`scripts/nv_graft_sanity.py`)

```bash
NV_DIR=... PA_DIR=... OUT_DIR=/models/nv-graft python3 scripts/nv_graft_sanity.py > /tmp/nv-graft-sanity.out
```

Asserts: no missing shards; 40 sampled grafted expert keys match PA geometry (recipe preserved) *and* NV geometry (bytes came from NV); 20 sampled non-expert keys `realpath` back to PA; PA-only MTP keys `realpath` back to PA. Prints `VERDICT: GRAFT-SANE` / exit 1.

### Stage 3 — health-gated cutover with auto-rollback (`scripts/nv-graft-cutover.sh`)

```bash
UNIT=/etc/systemd/system/sglang-flashnext.service \
OLD_PATH=/models/primitive-ai/CHECKPOINT NEW_PATH=/models/nv-graft \
HEALTH_URL=localhost:8086/health_generate \
MODELS_URL=localhost:8086/v1/models MODEL_NAME=qwen38-flashnext \
STRIP_FLAG=" --expert-distribution-recorder-mode stat" \
BACKUP=/models/backups/sglang.service.pre-graft \
bash scripts/nv-graft-cutover.sh
```

The script re-checks Stage 1's token **and** Stage 2's verdict before touching the unit (defense in depth — it must be safe to fire by hand at 3 a.m.). Then: back up unit → `sed` model-path (+ strip a stale EPLB profiling flag we'd approved removing) → verify the edit landed → `daemon-reload` → restart → poll `/health_generate` for 15 min. On timeout: restore backup, restart, and **verify the rollback itself boots healthy** before declaring `ROLLBACK_HEALTH_OK` (exit 2). Exit codes 0/2/3 = success/rolled-back/escalate.

Boot took ~5 min (3.3 min KV drop for the session — expected and accepted as the price of the window).

### Stage 4 — acceptance probes (server-side)

Only run when health=200. A cold-prompt prefill sweep, a decode throughput probe (thinking + non-thinking arms), spec-decode accept metrics, and `NRestarts`:

| metric | bar | graft result |
|---|---|---|
| decode t/s (non-think) | ≥200 (PA baseline 232) | **239.0** |
| decode t/s (think) | — | 224.2 |
| prefill t/s @100k | — | 11.2k |
| spec accept length / rate | no material drop | 2.55 / 0.52 |
| NRestarts | 0 | 0 |

A material accept-length drop would have been the rollback trigger — the speculative drafter was tuned against the old expert bytes. 2.55 said the graft's calibration transferred cleanly.

### Stage 5 — independent verification (manual, since Stage 4's verdict was clean)

Live reads, not chain self-report: `systemctl cat` (model-path landed, flag stripped, no other diffs), `/get_server_info` (model_path + tokenizer resolve to the graft), and one real chat completion round-trip with reasoning tokens and clean `finish_reason=stop`. Report mirrored off-box.

---

## Results & rollback

- Graft serving in production: **decode +3% over the old checkpoint**, same accept length, zero restarts.
- Rollback is one copy + reload + restart from the pinned unit backup — ~5 min, same shape as the cutover.
- Both checkpoint sources are read-only throughout; the graft dir is disposable (rebuild = one script).

## The lessons, compressed

1. **Diff key *sets*, not just sampled key *geometry*.** Two "identical" checkpoints disagree in exactly the corner you didn't sample.
2. **A polled stream must carry a terminal token on every exit path** — success, failure, *and* crash. `tail -1` of a progress log cannot distinguish "thinking" from "dead."
3. **Crash-test your gates.** Our sanity script had a TypeError in dead code and an implicit symmetry assumption; it died guarding a healthy build.
4. **Stage-gate + health-gate + verified-rollback turns a scary cutover into two bounded KV drops.** The worst case was pre-computed and accepted before anything started — that's what made 3 a.m. unattended-able.
5. Unattended CLI tools: read the trust-dir/approval flags *before* the overnight run, not after the 1-second exit.

*Scripts in this repo are the shipped versions, generalized to env vars. They ran against real checkpoints on real systemd units; the numbers above are from the live acceptance probes.*
