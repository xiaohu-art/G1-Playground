#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
python_bin="${G1_PLAYGROUND_PYTHON:-python}"
simulator_pid=""
policy_pid=""

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable '$python_bin' was not found; activate the project environment first" >&2
    exit 127
fi

usage() {
    cat <<'EOF'
Usage: bash scripts/run_sim_pipeline.sh [HYDRA_OVERRIDE ...]

Starts the Inspire MuJoCo simulator in the background and keeps the HOI policy
in the foreground. If motion.name is omitted, select it once with Up/Down and
Enter; the launcher passes that same motion to both processes.

Example:
  bash scripts/run_sim_pipeline.sh hoi=depth/smalltable
  bash scripts/run_sim_pipeline.sh hoi=depth/smalltable motion.name=sub17_smalltable_003_v02
EOF
}

stop_simulator() {
    if [[ -z "$simulator_pid" ]] || ! kill -0 "$simulator_pid" 2>/dev/null; then
        return
    fi
    kill -INT "$simulator_pid" 2>/dev/null || true
    for _ in {1..60}; do
        if ! kill -0 "$simulator_pid" 2>/dev/null; then
            wait "$simulator_pid" 2>/dev/null || true
            return
        fi
        sleep 0.05
    done
    kill -TERM "$simulator_pid" 2>/dev/null || true
    wait "$simulator_pid" 2>/dev/null || true
}

stop_policy() {
    if [[ -z "$policy_pid" ]] || ! kill -0 "$policy_pid" 2>/dev/null; then
        return
    fi
    kill -INT "$policy_pid" 2>/dev/null || true
    for _ in {1..60}; do
        if ! kill -0 "$policy_pid" 2>/dev/null; then
            wait "$policy_pid" 2>/dev/null || true
            return
        fi
        sleep 0.05
    done
    kill -TERM "$policy_pid" 2>/dev/null || true
    wait "$policy_pid" 2>/dev/null || true
}

stop_processes() {
    # Stop physics before the command publisher so the simulator watchdog does
    # not report an expected policy shutdown as a fault.
    stop_simulator
    stop_policy
}

select_motion() {
    local motion_output
    motion_output="$("$python_bin" -c '
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from g1_playground.utils import resolve_repo_path

with initialize_config_dir(version_base=None, config_dir=str(Path.cwd() / "configs")):
    cfg = compose(config_name="run_loco_hoi_track", overrides=sys.argv[1:])
with np.load(resolve_repo_path(cfg.motion.file), allow_pickle=False) as motions:
    for name in motions["motion_names"]:
        print(str(name))
' "$@")"
    mapfile -t motion_names <<<"$motion_output"
    if (( ${#motion_names[@]} == 0 )); then
        echo "No motions found for the selected HOI configuration" >&2
        return 1
    fi

    local index=0
    local key=""
    local suffix=""
    local count=${#motion_names[@]}
    while true; do
        printf '\r\033[2KSelect HOI motion with Up/Down, Enter confirms [%d/%d]: %s' \
            "$((index + 1))" "$count" "${motion_names[index]}" >&2
        IFS= read -rsn1 key
        case "$key" in
            "")
                printf '\n' >&2
                selected_motion="${motion_names[index]}"
                return
                ;;
            $'\x1b')
                suffix=""
                IFS= read -rsn2 -t 0.1 suffix || true
                if [[ "$suffix" == "[A" ]]; then
                    index=$(((index - 1 + count) % count))
                elif [[ "$suffix" == "[B" ]]; then
                    index=$(((index + 1) % count))
                fi
                ;;
        esac
    done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

cd "$repo_dir"

overrides=("$@")
motion_is_set=false
deployment_is_set=false
for override in "${overrides[@]}"; do
    case "$override" in
        motion.name=*) motion_is_set=true ;;
        deployment=sim) deployment_is_set=true ;;
        deployment=*)
            echo "run_sim_pipeline.sh only supports deployment=sim" >&2
            exit 2
            ;;
    esac
done

if [[ "$motion_is_set" == false ]]; then
    selected_motion=""
    select_motion "${overrides[@]}"
    overrides+=("motion.name=$selected_motion")
fi
overrides+=("motion.interactive=false")
if [[ "$deployment_is_set" == false ]]; then
    overrides+=("deployment=sim")
fi

mkdir -p logs
timestamp="$(date +%Y%m%d-%H%M%S)"
simulator_log="logs/simulate-$timestamp.log"

echo "Starting MuJoCo; simulator output: $simulator_log"
(
    trap - INT TERM
    exec "$python_bin" scripts/simulate.py --inspire "${overrides[@]}"
) >"$simulator_log" 2>&1 &
simulator_pid=$!
trap stop_processes EXIT INT TERM

for _ in {1..100}; do
    if ! kill -0 "$simulator_pid" 2>/dev/null; then
        echo "MuJoCo exited during startup. Last log lines:" >&2
        tail -40 "$simulator_log" >&2
        exit 1
    fi
    if [[ -e /dev/shm/g1_playground-depth.bin ]]; then
        break
    fi
    sleep 0.05
done

echo "Starting policy in this terminal"
(
    trap - INT TERM
    exec "$python_bin" scripts/loco_body_hand_pipeline.py "${overrides[@]}" </dev/tty >/dev/tty 2>&1
) &
policy_pid=$!
set +e
wait "$policy_pid"
policy_status=$?
set -e
exit "$policy_status"
