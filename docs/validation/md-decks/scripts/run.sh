#!/usr/bin/env bash
# Run selected materials sequentially; outputs remain in the caller's directory.
set -euo pipefail
: "${LAMMPS:=lmp}"
: "${OUT:?set OUT to an output directory}"
: "${MATERIALS:=cao,al2o3,tio2,mgo,cr2o3}"
: "${ATOMS:=1500}"; : "${EQ_STEPS:=50000}"; : "${PROD_STEPS:=150000}"
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
for material in ${MATERIALS//,/ }; do
  case "$material" in
    cao) density=2.90; tm=3200;; al2o3) density=2.93; tm=2327;;
    tio2) density=3.24; tm=2130;; mgo) density=2.75; tm=3105;;
    cr2o3) density=4.00; tm=2603;;
    sio2) density=2.20; tm=1986;;
    *) echo "unsupported material: $material" >&2; exit 2;;
  esac
  python3 "$root/scripts/make_cell.py" "$material" "$ATOMS" "$OUT/$material.data" --density "$density"
  "$LAMMPS" -in "$root/inputs/in.prep.lmp" -var data "$OUT/$material.data" -var outdata "$OUT/$material.prepared.data"
  for temp in "$tm" 2500 2473 2400 2300 2200; do
    "$LAMMPS" -in "$root/inputs/in.point.lmp" -var material "$material" -var load_restart 0 \
      -var data "$OUT/$material.prepared.data" -var temp "$temp" -var eq "$EQ_STEPS" \
      -var prod "$PROD_STEPS" -var samples "$OUT/$material-$temp.samples" -var restart "$OUT/$material-$temp.restart"
  done
done
