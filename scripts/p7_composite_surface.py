#!/usr/bin/env python
"""Phase-7 S3 — the COMPOSITE protein+ligand surface (a real neosurface). Runs INSIDE the .sif.

Phase 6C built each subunit's surface **without** the drug, so the pocket is represented as an empty
cavity and the composite shape a partner protein actually recognises never existed in our data at
all. That is a whole missing object, not a missing channel, and it is the leading suspect for the
axis-3 null. Here MSMS runs on the union of the subunit and its drug, so the drug fills the pocket
and contributes its own exposed face to one continuous surface.

**D7-4: inference only.** This is never a training input. Building a PDBbind protein surface with
its own ligand present would put the ligand's shape on both sides of the training pair and make
retrieval free (the same reasoning that made Path B two separate graphs in Phase 6C). PDBbind has no
partner protein, so there is no training data for a composite pair type.

The channel code is the reference's own ligand-aware branch — `computeCharges` and
`computeHydrophobicity` both route a vertex to the ligand helper when its residue name matches
`ligand_code`, which is exactly the mixed case here. Electrostatics use pdb2pqr for the protein plus
appended self-PQR rows for the ligand, then the shared `run_apbs`, so protein and ligand vertices are
sampled from ONE Poisson-Boltzmann solution.

Usage: p7_composite_surface.py --pdb <subunit.pdb> --sdf <drug.sdf> --out-prefix <dir/id>
"""
from __future__ import print_function

import argparse
import json
import os
import random
import sys
from subprocess import PIPE, Popen

import numpy as np
import pymesh
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p7_lig_surface import (DEFAULT_RADIUS, LIG_CHAIN, LIG_RESNAME, RADII,  # noqa: E402
                            load_mol, normalize_electrostatics, run_apbs, run_msms, shape_index,
                            write_pqr_rows)

from default_config.chemistry import polarHydrogens, radii as REF_RADII  # noqa: E402
from default_config.global_vars import pdb2pqr_bin  # noqa: E402
from default_config.masif_opts import masif_opts  # noqa: E402
from input_output.save_ply import save_ply  # noqa: E402
from triangulation.computeCharges import assignChargesToNewMesh, computeCharges  # noqa: E402
from triangulation.computeHydrophobicity import computeHydrophobicity  # noqa: E402
from triangulation.compute_normal import compute_normal  # noqa: E402
from triangulation.fixmesh import fix_mesh  # noqa: E402


def protein_atom_lines(pdb_path):
    """ATOM records of the (already protonated, already chain-extracted) subunit PDB."""
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6] in ("ATOM  ", "HETATM") and len(line) >= 54:
                out.append(line.rstrip("\n"))
    return out


def write_merged_pdb(prot_lines, mol, path):
    """Subunit ATOM records + the drug as HETATM LIG/chain L, in one file Bio.PDB can parse."""
    with open(path, "w") as fh:
        for line in prot_lines:
            fh.write(line + "\n")
        block = Chem.MolToPDBBlock(mol, flavor=4)
        serial = len(prot_lines)
        for line in block.split("\n"):
            if line[:6] not in ("ATOM  ", "HETATM"):
                continue
            serial += 1
            fh.write("HETATM%5d%s%s%4d%s\n"
                     % (serial, line[11:21], LIG_CHAIN, 1, line[26:]))
        fh.write("END\n")


def write_merged_xyzrn(prot_lines, mol, path):
    """Protein atoms under the reference's own rules; ligand atoms with the D7-2 radii."""
    n_prot = n_lig = 0
    with open(path, "w") as fh:
        for line in prot_lines:
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resid = line[22:26].strip()
            atomtype = name[0:1]
            # reference gate, verbatim: unknown element or non-standard residue -> skipped
            if atomtype not in REF_RADII or resname not in polarHydrogens:
                continue
            color = "Green"
            if atomtype == "O":
                color = "Red"
            elif atomtype == "N":
                color = "Blue"
            elif atomtype == "H" and name in polarHydrogens.get(resname, []):
                color = "Blue"
            fh.write("%s %s %s %s 1 %s_%s_x_%s_%s_%s\n"
                     % (line[30:38].strip(), line[38:46].strip(), line[46:54].strip(),
                        REF_RADII[atomtype], chain, resid, resname, name, color))
            n_prot += 1
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol().upper()
            r = RADII.get(sym, DEFAULT_RADIUS)
            nm = atom.GetPDBResidueInfo().GetName().strip()
            p = conf.GetAtomPosition(atom.GetIdx())
            color = {"O": "Red", "N": "Blue"}.get(sym, "Green")
            fh.write("%.06f %.06f %.06f %.6f 1 %s_%d_x_%s_%s_%s\n"
                     % (p.x, p.y, p.z, r, LIG_CHAIN, 1, LIG_RESNAME, nm, color))
            n_lig += 1
    return n_prot, n_lig


def merged_pqr(prot_pdb, mol, base):
    """pdb2pqr on the protein, then the ligand rows appended: one PQR, one PB solution."""
    directory = os.path.dirname(base) or "."
    stem = os.path.basename(base)
    args = [pdb2pqr_bin, os.path.abspath(prot_pdb), stem + "_prot.pqr",
            "--ff=PARSE", "--whitespace", "--noopt"]
    p = Popen(args, stdout=PIPE, stderr=PIPE, cwd=directory)
    _o, e = p.communicate()
    prot_pqr = os.path.join(directory, stem + "_prot.pqr")
    if not os.path.exists(prot_pqr):
        raise RuntimeError("pdb2pqr failed: %s" % e.decode("utf-8", "ignore")[-300:])
    out = base + ".pqr"
    n_prot = 0
    with open(out, "w") as fh:
        with open(prot_pqr) as src:
            for line in src:
                if line.startswith(("ATOM", "HETATM")):
                    fh.write(line if line.endswith("\n") else line + "\n")
                    n_prot += 1
        write_pqr_rows(mol, fh, start_serial=n_prot + 1, resnum=9999)
    os.remove(prot_pqr)
    return out


def build(args):
    rep = {"id": args.id or os.path.basename(args.out_prefix)}
    mol, err = load_mol(args.sdf, args.mol2)
    if mol is None:
        rep.update(ok=False, err=err)
        return rep
    prot_lines = protein_atom_lines(args.pdb)
    rep["n_prot_lines"] = len(prot_lines)
    rep["n_lig_atom"] = mol.GetNumAtoms()

    tmp = os.environ.get("TMPDIR", masif_opts["tmp_dir"])
    base = os.path.join(tmp, "p7cmp_%d" % random.randint(1, 10 ** 7))
    rep["n_xyzrn_prot"], rep["n_xyzrn_lig"] = write_merged_xyzrn(prot_lines, mol, base + ".xyzrn")
    merged_pdb = base + ".pdb"
    write_merged_pdb(prot_lines, mol, merged_pdb)

    verts1, faces1, normals1, names1 = run_msms(base)
    rep["n_msms_vert"] = len(verts1)
    lig_mask1 = np.array([nm.split("_")[3] == LIG_RESNAME for nm in names1])
    rep["msms_vert_ligand_owned"] = int(lig_mask1.sum())
    if rep["msms_vert_ligand_owned"] == 0:
        rep.update(ok=False, err="drug is fully buried: contributes no surface vertex")
        return rep

    hbond = computeCharges(base, verts1, names1, ligand_code=LIG_RESNAME, rdmol=mol)
    hphob = computeHydrophobicity(names1, ligand_code=LIG_RESNAME, rdmol=mol)

    mesh = pymesh.form_mesh(np.array(verts1), np.array(faces1))
    regular = fix_mesh(mesh, masif_opts["mesh_res"])
    rep["n_vert"] = int(regular.vertices.shape[0])
    vnorm = compute_normal(regular.vertices, regular.faces)
    hbond_r = assignChargesToNewMesh(regular.vertices, verts1, hbond, masif_opts)
    hphob_r = assignChargesToNewMesh(regular.vertices, verts1, hphob, masif_opts)

    charges = np.zeros(len(regular.vertices))
    try:
        pqr = merged_pqr(merged_pdb, mol, base)
        charges = run_apbs(pqr, regular.vertices, base)
        rep["apbs"] = "merged_pqr"
        os.remove(pqr)
    except Exception as exc:                                        # noqa: BLE001
        rep["apbs"] = "none"
        rep["apbs_err"] = str(exc)[-250:]

    si, n_clamped = shape_index(regular)
    feat = np.stack([si, hbond_r, normalize_electrostatics(charges / 10.0), hphob_r / 4.5],
                    axis=1).astype(np.float32)
    rep["si_clamped_frac"] = round(float(n_clamped) / max(len(si), 1), 4)

    # which regularised vertices are owned by the drug (nearest raw MSMS vertex carries the label)
    from scipy.spatial import cKDTree
    _d, nn = cKDTree(np.array(verts1)).query(regular.vertices, k=1)
    lig_mask = lig_mask1[nn]
    rep["vert_ligand_owned"] = int(lig_mask.sum())

    d = os.path.dirname(args.out_prefix)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    np.save(args.out_prefix + "_verts.npy", np.asarray(regular.vertices, dtype=np.float64))
    np.save(args.out_prefix + "_faces.npy", np.asarray(regular.faces, dtype=np.int64))
    np.save(args.out_prefix + "_normals.npy", np.asarray(vnorm, dtype=np.float64))
    np.save(args.out_prefix + "_feat.npy", feat)
    np.save(args.out_prefix + "_ligmask.npy", lig_mask.astype(np.int64))
    save_ply(args.out_prefix + ".ply", regular.vertices, regular.faces, normals=vnorm,
             charges=charges, normalize_charges=True, hbond=hbond_r, hphob=hphob_r)
    rep.update(ok=True, si=[float(si.min()), float(si.max()), float(si.mean())])
    for ext in (".xyzrn", ".vert", ".face", ".area", ".pdb"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="subunit chain PDB (protonated, extracted)")
    ap.add_argument("--sdf", required=True)
    ap.add_argument("--mol2", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--id", default=None)
    args = ap.parse_args()
    try:
        rep = build(args)
    except Exception as exc:                                        # noqa: BLE001
        import traceback
        if os.environ.get("P7_TRACE"):
            traceback.print_exc()
        rep = {"id": args.id, "ok": False, "err": "%s: %s" % (type(exc).__name__, exc)}
    print(json.dumps(rep))
    sys.exit(0 if rep.get("ok") else 1)


if __name__ == "__main__":
    main()
