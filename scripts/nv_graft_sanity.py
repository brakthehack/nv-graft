#!/usr/bin/env python3
"""Graft sanity gate: prove the built graft directory is loadable and correct.

Checks:
  1. every key in the graft index resolves to an existing shard file
  2. sampled expert tensors match primitive-ai geometry (dtype+shape) — the graft
     keeps the recipe — AND match NVIDIA geometry — the bytes came from NV
  3. non-expert shards are symlinks pointing at the primitive-ai originals
  4. PA-kept expert keys (no NV counterpart, e.g. fused MTP) are symlinked to PA,
     NOT accidentally swallowed by a graft part

Exit 0 + prints VERDICT: GRAFT-SANE iff all pass.
Config: NV_DIR, PA_DIR, OUT_DIR (same as build_nv_graft.py)
"""
import json, os, struct, random, sys

G = os.environ.get("OUT_DIR", "/models/nv-graft")
P = os.environ.get("PA_DIR", sys.exit("PA_DIR required"))
N = os.environ.get("NV_DIR", sys.exit("NV_DIR required"))

def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

gi = json.load(open(G + "/model.safetensors.index.json"))["weight_map"]
shards = sorted(set(gi.values()))
missing = [s for s in shards if not os.path.exists(os.path.join(G, s))]
print("graft keys:", len(gi), "| shards:", len(shards), "| missing:", missing)
if missing: sys.exit("FAIL missing shards")

hdr_cache = {}
def h(d, s):
    key = (d, s)
    if key not in hdr_cache:
        hdr_cache[key] = hdr(os.path.join(d, s))
    return hdr_cache[key]

pi = json.load(open(P + "/model.safetensors.index.json"))["weight_map"]
ni = json.load(open(N + "/model.safetensors.index.json"))["weight_map"]
# PITFALL #2 fix: sample only NV-comparable expert keys. PA fused-MTP expert keys
# are kept from PA by design (no NV counterpart) -> checking them "vs NV" is a
# guaranteed KeyError. Assert separately that they resolve via symlink to PA.
exp = [k for k in gi if ".mlp.experts." in k and k in ni]
pa_only = [k for k in gi if ".mlp.experts." in k and k not in ni]
print("PA-kept (no NV counterpart) expert keys:", len(pa_only))
assert all(os.path.realpath(os.path.join(G, gi[k])) == os.path.join(P, gi[k]) for k in pa_only), \
    "FAIL pa-only expert keys not symlinked to PA"
random.seed(6)
bad = 0
for k in random.sample(exp, min(40, len(exp))):
    g = h(G, gi[k])[k]; p = h(P, pi[k])[k]; n = h(N, ni[k])[k]
    if (g["dtype"], g["shape"]) != (p["dtype"], p["shape"]):
        print("GEOM-MISMATCH-VS-PA", k); bad += 1
    # data_offsets differ by layout by design; compare dtype+shape vs NV only.
    # (PITFALL #3: an earlier version tried to compare offsets too and crashed
    # the gate with a tuple+list TypeError — dead code with live consequences.)
    if (g["dtype"], g["shape"]) != (n["dtype"], n["shape"]):
        print("GEOM-MISMATCH-VS-NV", k); bad += 1
# non-expert keys must be byte-identical refs (symlinks to PA shards)
nonexp = [k for k in gi if ".mlp.experts." not in k]
nonexp_ok = all(os.path.realpath(os.path.join(G, gi[k])) == os.path.join(P, gi[k])
                for k in random.sample(nonexp, min(20, len(nonexp))))
print("sampled", min(40, len(exp)), "expert keys vs PA+NV geometry:", "OK" if bad == 0 else f"FAIL {bad}")
print("non-expert symlinks point at PA:", "OK" if nonexp_ok else "FAIL")
sane = bad == 0 and nonexp_ok and not missing
print("VERDICT:", "GRAFT-SANE" if sane else "GRAFT-BAD")
sys.exit(0 if sane else 1)
