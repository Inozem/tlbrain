#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$script_dir/bootstrap_runner.py"
image_metadata="$script_dir/image_metadata.py"
image_mode=false
verify_mode=false
if [[ "${1:-}" == "--image-mode" ]]; then
    image_mode=true
    shift
elif [[ "${1:-}" == "--verify-mode" ]]; then
    verify_mode=true
    shift
fi

is_store_alias() {
    case "${1//\\/\/}" in
        *[Ww][Ii][Nn][Dd][Oo][Ww][Ss][Aa][Pp][Pp][Ss]/*) return 0 ;;
        *) return 1 ;;
    esac
}

for candidate_name in python3; do
    if command -v "$candidate_name" >/dev/null 2>&1; then
        candidate="$(command -v "$candidate_name")"
        if ! is_store_alias "$candidate"; then
            if [[ "$image_mode" == true ]]; then
                exec "$candidate" "$image_metadata" image-preflight "$@"
            fi
            if [[ "$verify_mode" == true ]]; then
                exec "$candidate" "$image_metadata" verify "$@"
            fi
            exec "$candidate" "$runner" "$@"
        fi
    fi
done

if command -v powershell.exe >/dev/null 2>&1; then
    candidate="$(powershell.exe -NoLogo -NoProfile -NonInteractive -Command '$p = Get-Command -Name python.exe -CommandType Application -ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch "(?i)\\WindowsApps\\" } | Select-Object -First 1 -ExpandProperty Source; if (-not $p) { exit 1 }; [Console]::Out.WriteLine($p)' 2>/dev/null || true)"
    if [[ -n "$candidate" ]]; then
        if command -v cygpath >/dev/null 2>&1; then
            candidate="$(cygpath -u "$candidate")"
        fi
        if ! is_store_alias "$candidate"; then
            if [[ "$image_mode" == true ]]; then
                exec "$candidate" "$image_metadata" image-preflight "$@"
            fi
            if [[ "$verify_mode" == true ]]; then
                exec "$candidate" "$image_metadata" verify "$@"
            fi
            exec "$candidate" "$runner" "$@"
        fi
    fi
fi

if command -v gcloud >/dev/null 2>&1; then
    candidate="$(gcloud info --format='value(basic.python_location)' 2>/dev/null || true)"
    candidate="${candidate%$'\r'}"
    if [[ -n "$candidate" ]] && ! is_store_alias "$candidate"; then
        if command -v cygpath >/dev/null 2>&1; then
            candidate="$(cygpath -u "$candidate")"
        fi
        if ! is_store_alias "$candidate"; then
            if [[ "$image_mode" == true ]]; then
                exec "$candidate" "$image_metadata" image-preflight "$@"
            fi
            if [[ "$verify_mode" == true ]]; then
                exec "$candidate" "$image_metadata" verify "$@"
            fi
            exec "$candidate" "$runner" "$@"
        fi
    fi
fi

echo "preflight: no approved bootstrap interpreter found (provide Python or make gcloud available)" >&2
exit 1
