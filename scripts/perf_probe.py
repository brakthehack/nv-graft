#!/usr/bin/env python3
"""Flash-Next prefill/decode perf probe (OpenAI-compatible /v1/chat/completions).

Usage:
  BASE_URL=http://127.0.0.1:8086 MODEL=qwen38-flashnext TOKDIR=/models/tok python3 perf_probe.py

Requires: transformers (AutoTokenizer). No other deps.

Baselines it should roughly reproduce on a healthy single-GPU NVFP4 box: cold
prefill ~11K tok/s (flat 6K->100K), decode ~180-240 tok/s, warm TTFT ~0.15 s.

Key pitfalls baked in (do NOT remove):
  - tokenization via LOCAL AutoTokenizer (some forks' /encode returns HTTP 400)
  - cold prompts salted with a random prefix (radix/hierarchical cache would
    otherwise fake warm numbers on a busy box)
  - decode: ignore_eos=True + stream_options.include_usage; count
    content+reasoning_content (thinking tokens are real generated tokens)
  - occupancy sanity check first: never measure through someone else's generation
"""
import json, os, time, urllib.request, sys, random, string

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8086")
MODEL = os.environ.get("MODEL", "qwen38-flashnext")
TOKDIR = os.environ.get("TOKDIR", sys.exit("TOKDIR required: tokenizer dir"))

_TOK = None
def count_tokens(text):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])

def stream(body_extra, prompt, max_tokens=256, thinking=False):
    b = {"model": MODEL,
         "messages": [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": thinking}}
    b.update(body_extra)
    req = urllib.request.Request(URL + "/v1/chat/completions",
        data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; last = None; comp = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:"): continue
            p = s[5:].strip()
            if p == "[DONE]": break
            try: ev = json.loads(p)
            except Exception: continue
            if ev.get("usage"): comp = ev["usage"].get("completion_tokens")
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                c = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if c:
                    now = time.time() - t0
                    if first is None: first = now
                    last = now
    return first, last, comp

def prefill_run(tag, filler, reps):
    salt = "".join(random.choices(string.ascii_letters, k=12))
    prompt = salt + " " + filler * reps
    ptoks = count_tokens(prompt)
    first, last, comp = stream({"ignore_eos": True}, prompt, max_tokens=8)
    if first is None:
        print(json.dumps({"tag": tag, "error": "no streamed tokens"})); sys.stdout.flush(); return
    print(json.dumps({"tag": tag, "prompt_tokens": ptoks, "ttft_s": round(first, 2),
                      "prefill_tok_s": round(ptoks / first, 1)}))
    sys.stdout.flush()

def decode_run(tag, thinking=False):
    first, last, comp = stream({"ignore_eos": True},
        "Count from 1 to 2000, one number per line.", max_tokens=512, thinking=thinking)
    if first is None:
        print(json.dumps({"tag": tag, "error": "no streamed tokens"})); sys.stdout.flush(); return
    dur = last - first
    print(json.dumps({"tag": tag, "completion_tokens": comp, "ttft_s": round(first, 2),
                      "decode_tok_s": round((comp - 1) / dur, 1) if dur > 0 else None}))
    sys.stdout.flush()

if __name__ == "__main__":
    # occupancy sanity: don't measure through someone else's generation
    try:
        m = urllib.request.urlopen(URL + "/metrics", timeout=10).read().decode()
        run = [l for l in m.splitlines() if l.startswith("sglang:num_running_reqs")]
        print("occupancy:", run[0] if run else "?", file=sys.stderr)
    except Exception as e:
        print("metrics read failed:", e, file=sys.stderr)
    filler = ("The quick brown fox jumps over the lazy dog while parsing compiler "
              "intermediate representations. ")
    prefill_run("cold6k", filler, 400);   time.sleep(2)
    prefill_run("cold50k", filler, 3300); time.sleep(2)
    prefill_run("cold100k", filler, 6700); time.sleep(2)
    decode_run("decode-nonthink"); time.sleep(1)
    decode_run("decode-think", thinking=True)
    print("DONE")
