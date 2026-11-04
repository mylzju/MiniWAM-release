#!/usr/bin/env bash
# Fetch the exact DINOv3 assets from ModelScope for MiniWAM's offline loader.
set -euo pipefail

usage() {
    printf 'Usage: bash scripts/download_dinov3_modelscope.sh [--check-only]\n'
}

check_only=0
case "${1:-}" in
    --check-only) check_only=1 ;;
    -h|--help) usage; exit 0 ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID='facebook/dinov3-vitb16-pretrain-lvd1689m'
MODELSCOPE_REVISION='23d0280ae6ee4ced592a3459674ad027d3c18906'
# These byte-identical assets retain the revision recorded in existing configs.
OFFLINE_REVISION='5931719e67bbdb9737e363e781fb0c67687896bc'
CACHE_ROOT="${HF_HUB_CACHE:-$ROOT_DIR/pretrained/hf-cache}"
MODEL_CACHE="$CACHE_ROOT/models--facebook--dinov3-vitb16-pretrain-lvd1689m"
SNAPSHOT="$MODEL_CACHE/snapshots/$OFFLINE_REVISION"

matches_checksum() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status 2>/dev/null
}

if [[ "$check_only" -eq 0 ]]; then
    mkdir -p "$SNAPSHOT"
fi

while read -r expected filename; do
    target="$SNAPSHOT/$filename"
    if matches_checksum "$expected" "$target"; then
        printf 'Verified %s\n' "$filename"
        continue
    fi
    if [[ "$check_only" -eq 1 ]]; then
        printf 'Missing or incorrect DINOv3 asset: %s\n' "$target" >&2
        exit 1
    fi

    partial="$target.incomplete"
    if ! matches_checksum "$expected" "$partial"; then
        curl --fail --location --retry 3 --retry-delay 2 \
            --connect-timeout 20 --max-time 1800 --continue-at - \
            "https://modelscope.cn/api/v1/models/$MODEL_ID/repo?Revision=$MODELSCOPE_REVISION&FilePath=$filename" \
            --output "$partial"
    fi
    if ! matches_checksum "$expected" "$partial"; then
        printf 'SHA256 mismatch: %s; rename this incomplete file before retrying.\n' "$partial" >&2
        exit 1
    fi
    mv -- "$partial" "$target"
    printf 'Downloaded and verified %s\n' "$filename"
done <<'CHECKSUMS'
3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec config.json
960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7 preprocessor_config.json
9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b model.safetensors
25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e LICENSE.md
42990cce5247479866ce2e6aa375eb3c3d1717966c41b74890d377a31ad0b0fe README.md
CHECKSUMS

if [[ "$check_only" -eq 0 ]]; then
    mkdir -p "$MODEL_CACHE/refs"
    printf '%s' "$OFFLINE_REVISION" > "$MODEL_CACHE/refs/main"
fi
printf 'Offline DINOv3 snapshot verified: %s\n' "$SNAPSHOT"
