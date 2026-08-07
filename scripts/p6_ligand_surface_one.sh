#!/bin/bash
# Build a LIGAND-MODIFIED surface for one PDBbind complex (protein + ligand) via the .sif.
# Merges PDBbind {id}_protein.pdb + {id}_ligand.sdf (as HETATM LIG chain X) -> raw PDB, then runs
# 01-triangulate with ligand args so MSMS includes the ligand. Usage: p6_ligand_surface_one.sh <pdbbind_id> <prot_chain>
set -u
ID="$1"; PCH="${2:-A}"; LIGC="LIG"; LCH="X"
REFROOT=/scratch/ymeng/masif-graph/masif-neosurf-af2
REFDATA=$REFROOT/masif/data/masif_ppi_search
SRC=$REFROOT/masif/source
SIF=$REFROOT/masif-neosurf_v0.1.sif
RAW=$REFDATA/data_preparation/00-raw_pdbs
PLB=/scratch/ymeng/masif-graph/data/pdbbind/$ID
PY=/work/upthomae/Meng/conda_envs/masif-graph/bin/python
LOG=/scratch/ymeng/masif-graph/logs/phase6/ligsurf_${ID}.log
mkdir -p "$RAW" logs/phase6
# 1) ligand sdf -> HETATM PDB block (LIG / chain X), merge with protein.pdb -> raw {ID}.pdb
PYTHONPATH=/scratch/ymeng/masif-graph/src $PY - "$ID" "$PCH" "$LIGC" "$LCH" <<'PY'
import sys
from rdkit import Chem
ID,PCH,LIGC,LCH=sys.argv[1:5]
base=f"/scratch/ymeng/masif-graph/data/pdbbind/{ID}"
m=Chem.SDMolSupplier(f"{base}/{ID}_ligand.sdf", removeHs=False)[0]
m=Chem.AddHs(m, addCoords=True)
for a in m.GetAtoms():
    ri=Chem.AtomPDBResidueInfo(); ri.SetResidueName(f"{LIGC:>3}"); ri.SetChainId(LCH)
    ri.SetResidueNumber(1); ri.SetIsHeteroAtom(True)
    nm=a.GetSymbol(); ri.SetName(f" {nm:<3}"[:4]); a.SetMonomerInfo(ri)
lig=Chem.MolToPDBBlock(m, flavor=0)
lighet=[l for l in lig.splitlines() if l.startswith(("HETATM","ATOM"))]
lighet=[("HETATM"+l[6:]) for l in lighet]
prot=[l for l in open(f"{base}/{ID}_protein.pdb") if l.startswith(("ATOM","TER"))]
raw=f"/scratch/ymeng/masif-graph/masif-neosurf-af2/masif/data/masif_ppi_search/data_preparation/00-raw_pdbs/{ID}.pdb"
with open(raw,"w") as f:
    f.writelines(prot); f.write("TER\n"); f.write("\n".join(lighet)+"\n"); f.write("END\n")
print(f"raw PDB: {sum(1 for l in prot if l.startswith('ATOM'))} protein atoms + {len(lighet)} ligand HETATM -> {raw}")
PY
export PYTHONPATH="${PYTHONPATH:-}:$SRC:$REFDATA"
SEXEC="singularity exec --bind $REFROOT:$REFROOT --bind /work:/work --bind /scratch/ymeng/masif-graph:/scratch/ymeng/masif-graph $SIF python"
cd "$REFDATA"
{
echo "=== LIGSURF $ID chain $PCH + $LIGC $(date '+%T') ==="
echo "--- 01 triangulate (WITH ligand $LIGC _ $LCH) ---"
timeout 1800 $SEXEC "$SRC/data_preparation/01-pdb_extract_and_triangulate.py" "${ID}_${PCH}" "${LIGC}_${LCH}" "$PLB/${ID}_ligand.sdf"; echo "  rc=$?"
ls -la data_preparation/01-benchmark_surfaces/${ID}_${PCH}.ply 2>/dev/null && echo "  PLY_OK" || echo "  PLY_MISSING"
} >> "$LOG" 2>&1
grep -E 'rc=|PLY_|Including ligand|Error|Traceback' "$LOG" | tail -8
