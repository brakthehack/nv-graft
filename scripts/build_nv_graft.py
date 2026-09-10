#!/usr/bin/env python3
"""Build a graft checkpoint: primitive-ai skeleton + NVIDIA-calibrated NVFP4 expert tensors.

Motivation: the NVIDIA-republished NVFP4 checkpoint carries re-calibrated expert
scales but ships its MTP drafter layer in a different (per-expert) tensor layout
that our serving fork's NEXTN path doesn't parse. The primitive-ai checkpoint has
the exact dialect the fork loads (fused BF16 MTP + the right quantization ignore
list) but older expert calibration. A "graft" takes the primitive-ai skeleton
config + non-expert tensors and swaps in ONLY the NVIDIA expert tensors, where
key geometry (dtype/shape/scale triplet) is byte-identical between the two.

Output: $OUT_DIR — a symlink-farm over the primitive-ai directory. Non-expert
shards are symlinks (zero copy, read-only on both sources); expert tensors are
repacked into nv-experts-partNN.safetensors; the index is rewritten.

Read-only on both sources. CPU/disk only, ~65 GB output.
Run with a python that has safetensors + torch installed.

SUPPORTED MODELS: tested only on the nvidia/ + primitive-ai builds of
Qwen3.8-Flash-Next-NVFP4. The graft scope below (".mlp.experts." key matching,
PA-only fused-MTP handling) is specific to that pair; on any other checkpoint
this is untested code -- adapt scope and pass the sanity gate before serving.

Config:
  NV_DIR   (required) NVIDIA NVFP4 checkpoint dir
  PA_DIR   (required) primitive-ai NVFP4 checkpoint dir
  OUT_DIR  (default /models/nv-graft)
  LOG      (default /tmp/nv-graft-build.log)
"""
import json, os, sys, collections, struct, re
from safetensors.torch import load_file, save_file

N = os.environ.get("NV_DIR", sys.exit("NV_DIR required: nvidia NVFP4 checkpoint dir"))
P = os.environ.get("PA_DIR", sys.exit("PA_DIR required: primitive-ai NVFP4 checkpoint dir"))
OUT = os.environ.get("OUT_DIR", "/models/nv-graft")
LOG = os.environ.get("LOG", "/tmp/nv-graft-build.log")

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); open(LOG, "a").write(msg + "\n")

idx_n = json.load(open(N + "/model.safetensors.index.json"))["weight_map"]
idx_p = json.load(open(P + "/model.safetensors.index.json"))["weight_map"]

def layer_of(k):
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None

# GRAFT SCOPE = intersection of expert keys present in BOTH indices.
# PITFALL #1 (the one that killed our first overnight run): PA's fused-MTP expert
# keys (mtp.layers.0.mlp.experts.{gate_up,down}_proj) have NO NVIDIA counterpart
# (NV ships 3072 per-expert mtp.* keys instead). Selecting scope from the PA
# index alone gives KeyError on the first NV lookup. The 2 PA-only keys stay
# from PA via the symlink farm — they are exactly the keys we're keeping PA for.
expert_keys_p = {k: v for k, v in idx_p.items() if ".mlp.experts." in k and k in idx_n}
by_layer = collections.defaultdict(list)
for k in expert_keys_p:
    by_layer[layer_of(k)].append(k)
log(f"graft scope: {len(expert_keys_p)} expert keys across {len(by_layer)} layers")

# verify NV geometry matches PA for sample keys BEFORE any heavy work
def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))
pa_hdr_cache, nv_hdr_cache = {}, {}
def get_hdr(d, shard, cache):
    if shard not in cache:
        cache[shard] = hdr(os.path.join(d, shard))
    return cache[shard]
mismatch = []
for k in list(expert_keys_p)[:60] + [c for c in expert_keys_p if c.endswith(".weight")][:60]:
    hp = get_hdr(P, idx_p[k], pa_hdr_cache)[k]
    hn_ = get_hdr(N, idx_n[k], nv_hdr_cache)[k]
    if (hp["dtype"], hp["shape"]) != (hn_["dtype"], hn_["shape"]):
        mismatch.append((k, hp["dtype"], hp["shape"], hn_["dtype"], hn_["shape"]))
log("geometry sample check:", "PASS" if not mismatch else f"FAIL {mismatch[:5]}")
if mismatch: sys.exit(1)
del pa_hdr_cache, nv_hdr_cache  # free

# extract NV expert tensors, one NVIDIA shard at a time (memory-safe: ~10 GB/shard)
os.makedirs(OUT, exist_ok=True)
nv_shards = sorted({idx_n[k] for k in expert_keys_p})
log(f"reading {len(nv_shards)} nvidia shards (experts only, rest discarded)")
part_map = {}  # nv shard -> graft part filename
for i, sh in enumerate(nv_shards):
    want = {k for k in expert_keys_p if idx_n[k] == sh}
    if not want: continue
    full = load_file(os.path.join(N, sh))
    kept = {k: full[k] for k in want}
    del full
    out = f"nv-experts-part{i:02d}.safetensors"
    save_file(kept, os.path.join(OUT, out), metadata={"format": "pt"})
    part_map[sh] = out
    kept.clear(); del kept
    log(f"  {sh}: wrote {out}")

# symlink farm over PA (only shards NOT replaced by graft parts) + new index
new_idx = {}
for k, v in idx_p.items():
    if k in expert_keys_p:
        new_idx[k] = part_map[idx_n[k]]
    else:
        new_idx[k] = v
        src = os.path.join(P, v)
        dst = os.path.join(OUT, v)
        if not os.path.lexists(dst):  # lexists: a BROKEN symlink from a stale build still occupies the name
            os.symlink(src, dst)
json.dump({"metadata": {"total_size": json.load(open(P + "/model.safetensors.index.json"))["metadata"]["total_size"]}, "weight_map": new_idx},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
log(f"index: {len(new_idx)} keys")
# non-shard metadata files (config, tokenizer, chat template...): symlink from PA
for fn in os.listdir(P):
    if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn == ".cache":
        continue
    dst = os.path.join(OUT, fn)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(P, fn), dst)
# terminal gate: every shard referenced by the index must exist
missing = [s for s in set(new_idx.values()) if not os.path.exists(os.path.join(OUT, s))]
log("missing shards:", missing)
log("GRAFT_BUILD_DONE" if not missing else "GRAFT_BUILD_FAIL")
sys.exit(0 if not missing else 1)
