#!/bin/bash
# Union symlink dir so p4.train (single --data dir) can see the whole combined corpus.
set -eu
P6=/work/upthomae/Meng/phase6C
mkdir -p "$P6/npz_all"
find "$P6/npz_all" -xtype l -delete 2>/dev/null || true
for d in npz_ppi npz_pl npz_eval; do
  [ -d "$P6/$d" ] || continue
  for f in "$P6/$d"/*.npz; do
    b=$(basename "$f"); [ -e "$P6/npz_all/$b" ] || ln -s "$f" "$P6/npz_all/$b"
  done
done
echo "npz_all: $(ls "$P6/npz_all" | wc -l) entries"
