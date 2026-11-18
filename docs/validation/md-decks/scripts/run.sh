#!/usr/bin/env bash
# Run selected materials sequentially; outputs remain in the caller's directory.
set -euo pipefail
: "${LAMMPS:=lmp}"
: "${OUT:?set OUT to an output directory}"
: "${MATERIALS:=cao,al2o3,tio2,mgo,cr2o3}"
: "${ATOMS:=1500}"; : "${EQ_STEPS:=50000}"; : "${PROD_STEPS:=150000}"
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
out_dir=$(cd "$OUT" && pwd)
manifest="$out_dir/progress.json"
python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "sampling": {
        "cell_seed": 1216,
        "langevin_seed": 48279,
        "langevin_streams_per_point": 1,
    },
    "points": [],
}, indent=2) + "\n")
PY
for material in ${MATERIALS//,/ }; do
  case "$material" in
    cao) density=2.90; tm=3200;; al2o3) density=2.93; tm=2327;;
    tio2) density=3.24; tm=2130;; mgo) density=2.75; tm=3105;;
    cr2o3) density=4.00; tm=2603;;
    sio2) density=2.20; tm=1986;;
    *) echo "unsupported material: $material" >&2; exit 2;;
  esac
  python3 "$root/scripts/make_cell.py" "$material" "$ATOMS" "$out_dir/$material.data" --density "$density"
  "$LAMMPS" -in "$root/inputs/in.prep.lmp" -var data "$out_dir/$material.data" -var outdata "$out_dir/$material.prepared.data"
  for temp in "$tm" 2500 2473 2400 2300 2200; do
    samples="$out_dir/$material-$temp.samples"
    "$LAMMPS" -in "$root/inputs/in.point.lmp" -var material "$material" -var load_restart 0 \
      -var data "$out_dir/$material.prepared.data" -var temp "$temp" -var eq "$EQ_STEPS" \
      -var prod "$PROD_STEPS" -var samples "$samples" -var restart "$out_dir/$material-$temp.restart"
    test -s "$samples"
    python3 - "$manifest" "$material" "$temp" "$out_dir/$material.prepared.data" \
      "$samples" "$ATOMS" "$PROD_STEPS" "$root/analysis" <<'PY'
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[8])
from analyze import OXIDES

manifest_path = Path(sys.argv[1])
material = sys.argv[2]
temp_K = int(sys.argv[3])
data_path = Path(sys.argv[4])
samples_path = Path(sys.argv[5])
requested_atoms = int(sys.argv[6])
production_steps = int(sys.argv[7])
atom_count, _formula_mass = OXIDES[material]
matches = re.findall(r"^\s*(\d+)\s+atoms\s*$", data_path.read_text(), re.MULTILINE)
if len(matches) != 1:
    raise SystemExit(f"could not read actual atom count from {data_path}")
actual_atoms = int(matches[0])
if actual_atoms % atom_count:
    raise SystemExit(f"{material} cell has a non-stoichiometric atom count: {actual_atoms}")
formula_units = actual_atoms // atom_count
expected_formula_units = max(1, round(requested_atoms / atom_count))
if formula_units != expected_formula_units:
    raise SystemExit(f"{material} cell formula units disagree with the requested atom count")
manifest = json.loads(manifest_path.read_text())
manifest["points"].append({
    "material": material,
    "temp_K": temp_K,
    "status": "complete",
    "samples": str(samples_path.resolve()),
    "requested_atoms": requested_atoms,
    "actual_atoms": actual_atoms,
    "formula_units": formula_units,
    "production_steps": production_steps,
    "sample_interval": 100,
})
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  done
done
