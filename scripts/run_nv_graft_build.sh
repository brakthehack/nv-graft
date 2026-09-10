#!/bin/bash
# Fail-safe wrapper around build_nv_graft.py.
# PITFALL #4 (cost us a 3-hour dead wait): the build's poller tails the LOG file,
# but python tracebacks go to stderr in a separate file. A crash left the log
# frozen on its last progress line and the supervisor waited out the full
# timeout assuming "still building". Rule: any non-zero exit MUST write a terminal
# verdict token into the polled log.
set -uo pipefail
LOG="${LOG:-/tmp/nv-graft-build.log}"
OUT="${BUILD_OUT:-/tmp/nv-graft-build.out}"
# Resolve relative to this wrapper's own dir: the README runs us as
# `bash scripts/run_nv_graft_build.sh` from the repo root, where a bare
# "build_nv_graft.py" does not exist in the cwd (this bit us: rc=2, instant FAIL).
python3 "${SCRIPT:-$(dirname "$0")/build_nv_graft.py}" >> "$OUT" 2>&1
rc=$?
# Write FAIL on ANY non-zero exit, unconditionally. Do NOT suppress it when the
# log already contains a GRAFT_BUILD_* token: that token may be STALE from a
# previous successful run, and a rebuild that crashes would then leave the log
# ending in the old DONE — gate 1 of the cutover would wave through a broken dir.
# Double FAIL tokens are harmless (cutover greps, doesn't parse the tail count).
if [ $rc -ne 0 ] && ! tail -1 "$LOG" 2>/dev/null | grep -q GRAFT_BUILD_FAIL; then
  echo "graft build crashed rc=$rc (see $OUT)" >> "$LOG"
  echo "GRAFT_BUILD_FAIL" >> "$LOG"
fi
exit $rc
