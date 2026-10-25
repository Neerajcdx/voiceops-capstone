#!/usr/bin/env bash
# Graded run command: ./run.sh <folder_of_audio> [--out out]
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run.py "$@"
