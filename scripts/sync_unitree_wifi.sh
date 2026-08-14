#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/sync_unitree_wifi.sh [--apply] [--no-delete]

Mirror this G1-Playground checkout to unitree-wifi:~/G1-Playground/.

  --apply      Perform the transfer. Without this flag, only show a dry run.
  --no-delete  Keep remote source files that no longer exist locally.
  -h, --help   Show this help.

Remote .git, environments, native libraries, build outputs, caches, and logs are preserved.
Generated third_party/*.egg-info directories are removed when deletion is enabled.
EOF
}

apply=false
delete=true
while (($#)); do
    case "$1" in
        --apply)
            apply=true
            ;;
        --no-delete)
            delete=false
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

command -v rsync >/dev/null 2>&1 || {
    echo "rsync is required but was not found in PATH" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
remote="unitree-wifi"
destination="~/G1-Playground/"

rsync_args=(
    -rlpt
    --human-readable
    --itemize-changes
    --exclude=.git/
    --exclude=.ruff_cache/
    --exclude=__pycache__/
    --exclude=.pytest_cache/
    --exclude=.mypy_cache/
    --exclude=.cache/
    --exclude=.venv/
    --exclude=venv/
    --exclude=build/
    --exclude=dist/
    --exclude=/g1_playground.egg-info/
    --filter='-s /third_party/**/*.egg-info/***'
    --exclude=logs/
    --exclude='*.py[cod]'
    --exclude='*.so'
    --exclude='*.o'
    --exclude='*.a'
    --exclude='*.whl'
    --exclude='*.log'
)

if $delete; then
    # Excluded files are protected unless --delete-excluded is used; never add it.
    rsync_args+=(--delete-delay)
fi
if ! $apply; then
    rsync_args+=(--dry-run)
fi

echo "Source:      $repo_root/"
echo "Destination: $remote:$destination"
if ! $apply; then
    echo "Mode:        dry run (re-run with --apply to transfer)"
else
    echo "Mode:        apply"
fi

rsync "${rsync_args[@]}" "$repo_root/" "$remote:$destination"
