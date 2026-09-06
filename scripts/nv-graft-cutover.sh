#!/bin/bash
# Health-gated model cutover with automatic rollback.
# Swaps a systemd unit's --model-path to the graft dir (plus any one-off flag
# cleanup you pass via STRIP_FLAG), restarts, polls health for up to 15 min.
# On health timeout it restores the backed-up unit and VERIFIES the rollback
# boots before declaring anything. Exit codes: 0 = success, 2 = rolled back
# (old model healthy), 3 = rollback also failed (escalate).
#
# Config (env):
#   UNIT        systemd unit path              (default /etc/systemd/system/sglang-flashnext.service)
#   OLD_PATH    model-path string to replace   (default the primitive-ai dir)
#   NEW_PATH    graft dir                      (default /models/nv-graft)
#   BACKUP      rollback unit copy destination (default /models/backups/sglang.service.pre-graft)
#   BUILD_LOG   build log to gate on           (default /tmp/nv-graft-build.log)
#   SANITY_OUT  sanity output to gate on       (default /tmp/nv-graft-sanity.out)
#   HEALTH_URL  e.g. localhost:8086/health_generate  (required)
#   MODELS_URL  e.g. localhost:8086/v1/models        (optional serve-name check)
#   MODEL_NAME  served model name expected in MODELS_URL output
#   STRIP_FLAG  optional flag substring to remove from ExecStart (e.g. profiling cruft)
#   SUDO        prefix for unit edits           (default sudo; set '' if already root)
set -uo pipefail
UNIT="${UNIT:?}"
OLD_PATH="${OLD_PATH:?}"; NEW_PATH="${NEW_PATH:?}"; HEALTH_URL="${HEALTH_URL:?}"
BACKUP="${BACKUP:-/models/backups/sglang.service.pre-graft}"
BUILD_LOG="${BUILD_LOG:-/tmp/nv-graft-build.log}"
SANITY_OUT="${SANITY_OUT:-/tmp/nv-graft-sanity.out}"
LOG="${CUTOVER_LOG:-/tmp/nv-graft-cutover.log}"
SUDO="${SUDO:-sudo}"
say(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "CUTOVER_START"

# gate 1: build finished with its terminal token
tail -1 "$BUILD_LOG" | grep -q GRAFT_BUILD_DONE || { say "ABORT build not done"; exit 1; }
# gate 2: sanity verdict passed
grep -q "VERDICT: GRAFT-SANE" "$SANITY_OUT" || { say "ABORT sanity not run/passed"; exit 1; }

$SUDO cp "$UNIT" "$BACKUP" && say "unit backed up to $BACKUP"
$SUDO sed -i "s|$OLD_PATH|$NEW_PATH|" "$UNIT"
[ -n "${STRIP_FLAG:-}" ] && $SUDO sed -i "s|${STRIP_FLAG}||" "$UNIT"
grep -q "$NEW_PATH" "$UNIT" || { say "ABORT sed did not land"; $SUDO cp "$BACKUP" "$UNIT"; exit 1; }
say "unit edited: graft path${STRIP_FLAG:+ + flag stripped}"
$SUDO systemctl daemon-reload
sleep 40
$SUDO systemctl restart "$(basename "$UNIT")"
say "RESTARTED"
for i in $(seq 1 60); do   # 60 x 15s = 15 min boot budget
  sleep 15
  H=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$HEALTH_URL" 2>/dev/null)
  [ "$H" = "200" ] && { say "HEALTH_OK poll=$i"
    if [ -n "${MODELS_URL:-}" ] && [ -n "${MODEL_NAME:-}" ]; then
      curl -s -m 10 "$MODELS_URL" | grep -q "$MODEL_NAME" && say "MODELS_OK"
    fi
    say "CUTOVER_SUCCESS"; exit 0; }
  [ $((i % 10)) -eq 0 ] && say "poll $i health=$H"
done
say "HEALTH_TIMEOUT -> ROLLBACK"
$SUDO cp "$BACKUP" "$UNIT"
$SUDO systemctl daemon-reload && $SUDO systemctl restart "$(basename "$UNIT")"
for i in $(seq 1 60); do
  sleep 15
  H=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$HEALTH_URL" 2>/dev/null)
  [ "$H" = "200" ] && { say "ROLLBACK_HEALTH_OK poll=$i"; exit 2; }
done
say "ROLLBACK_ALSO_FAILED_ESCALATE"
exit 3
