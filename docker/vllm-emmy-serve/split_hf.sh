#!/bin/bash
# Split warm/hf into warm/hf_parts/p0..p23 for the bake. Docker Hub rejects very large
# blobs (the monolithic ~24 GB snapshot layer 503s at upload initiation, forever), so
# the Dockerfile COPYs 24 sub-10 GB parts that merge back into /opt/emmy/hf — room for a
# 156 GB checkpoint; a small one leaves most parts empty. Split is by hardlink (no extra
# disk); symlinks (the hub snapshots/ -> blobs/ links) stay symlinks. Everything except
# the big blobs/ payloads goes to p0; the blob files are balanced across all parts,
# largest first. The native image's bake splits its pack the same way: `split_hf.sh SRC DST`.
set -euo pipefail
cd "$(dirname "$0")"
SRC=${1:-warm/hf}
DST=${2:-warm/hf_parts}

rm -rf "$DST"
PARTS=24  # the Dockerfile's COPY count
# A payload is a file over 64 MiB: a native pack's page files are 256 MiB each, which `-size +256M` (over 256 MiB)
# leaves in p0.
BIG=+64M
for i in $(seq 0 $((PARTS - 1))); do mkdir -p "$DST/p$i"; done

# p0 = the full tree minus the large payloads (configs, tokenizer, refs, symlinks).
# Split by SIZE, not by path: depending on how the snapshot was produced, the weight
# shards can sit under blobs/ or as regular files under snapshots/.
cp -al "$SRC"/. "$DST"/p0/
find "$DST"/p0 -type f -size "$BIG" -delete

# balance the large payloads across the parts, largest first into the emptiest part.
# The list is captured in a plain assignment (NOT a `< <(...)` process substitution, whose
# failure `set -e` cannot see): with pipefail, a failing find — e.g. BSD find without
# -printf on a macOS host — aborts the script here instead of silently yielding an empty
# loop with every weight shard already deleted from p0 above.
big=$(find "$SRC" -type f -size "$BIG" -printf "%s %p\n" | sort -rn)
declare -a used=()
for i in $(seq 0 $((PARTS - 1))); do used[i]=0; done
if [ -n "$big" ]; then
    while read -r sz f; do
        rel=${f#"$SRC"/}
        best=0
        for i in $(seq 1 $((PARTS - 1))); do [ "${used[$i]}" -lt "${used[$best]}" ] && best=$i; done
        mkdir -p "$DST/p$best/$(dirname "$rel")"
        ln "$f" "$DST/p$best/$rel"
        used[$best]=$((used[$best] + sz))
    done <<< "$big"
fi

# The split must be COMPLETE (every source file lands in exactly one part — a dropped
# shard bakes a weightless image that only dies ~30 min later in verify) and every part
# must stay under Docker Hub's ~10 GB blob cap (the exact 503 this mechanism exists to
# avoid — reshard_snapshot.py only rewrites model.safetensors, so any other >10 GB file
# would otherwise ride through into one part).
count() { find "$1" -type f -printf "%s\n" | awk '{n+=1; s+=$1} END {printf "%d %d", n, s+0}'; }
read -r src_n src_bytes <<< "$(count "$SRC")"
read -r dst_n dst_bytes <<< "$(count "$DST")"
if [ "$src_n" -ne "$dst_n" ] || [ "$src_bytes" -ne "$dst_bytes" ]; then
    echo "[split] FAIL: parts hold $dst_n files / $dst_bytes bytes, source has $src_n / $src_bytes" >&2
    exit 1
fi
for i in $(seq 0 $((PARTS - 1))); do
    p=p$i
    read -r _ pbytes <<< "$(count "$DST/$p")"
    if [ "$pbytes" -ge 10000000000 ]; then
        echo "[split] FAIL: $p is $pbytes bytes — over Docker Hub's ~10 GB blob cap (re-shard the payload first)" >&2
        exit 1
    fi
done

echo "[split] $(du -sh "$DST"/p* | tr '\n' ' ')"
