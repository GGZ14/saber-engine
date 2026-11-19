#!/usr/bin/env bash
# Console helpers shared by the launchers (docs/CONSOLE.md).
#
#   . "$REPO/tools/lib/console.sh"
#   console_stop       # stop a running Saber console (`saber ui`) the way Ctrl-C does
#
# `saber ui` is a host process, not a container, so a launcher's `stop` that only removes the engine
# container leaves the console (and its port) behind. SIGTERM is the console's own clean exit: it
# stops the engine it started, then exits (the same path as Ctrl-C). Prints what it did; returns 0
# when no console is left running, 1 when one ignored SIGTERM for 30 s (it is never SIGKILLed: it
# may be draining requests).
console_stop() {
  local pids p i
  pids=$(pgrep -u "$(id -u)" -f -- '/saber ui( |$)' 2>/dev/null) || { echo "no console was running"; return 0; }
  for p in $pids; do kill -TERM "$p" 2>/dev/null && echo "stopping the console (pid $p)"; done
  for i in $(seq 1 60); do
    pgrep -u "$(id -u)" -f -- '/saber ui( |$)' >/dev/null 2>&1 || { echo "stopped the console"; return 0; }
    sleep 0.5
  done
  echo "the console (pid ${pids//$'\n'/ }) is still draining after 30 s; it was not killed (kill -9 it if it is stuck)" >&2
  return 1
}
