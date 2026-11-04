#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$task_root/scripts/run_loft_pced_sentence_pass.sh" --reasoning-style evidence_v2 "$@"
