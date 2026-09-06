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
python3 "${SCRIPT:-build_nv_graft.py}" >> "$OUT" 2>&1
rc=$?
if [ $rc -ne 0 ] && ! grep -q GRAFT_BUILD_ "$LOG"; then
  echo "graft build crashed rc=$rc (see $OUT)" >> "$LOG"
  echo "GRAFT_BUILD_FAIL" >> "$LOG"
fi
exit $rc
