#!/usr/bin/env python
"""Phase-7 S0/S2 — build an MSMS molecular surface for a LIGAND ALONE (runs INSIDE the .sif).

Design D7-1/D7-2: reuse the reference pipeline rather than write a new one, so the ligand surface is
the same *kind of object* as a protein chain surface — same probe (1.5 A), same density (3.0), same
`fix_mesh(mesh_res=1.0)`, same four channels with the same normalisations.

What this replaces is `triangulation/ligand_utils.py::extract_ligand`, whose only job is to hand back
`(rdmol, mol2)` and which fails its own protonation assertion after re-deriving connectivity from PDB
HETATM records. We already have that molecule: PDBbind ships `{id}_ligand.sdf` with correct bond
orders in the bound pose, and the ternary benchmark gets bound-pose SDFs from the RCSB ModelServer.

Two deliberate divergences from the reference, both documented in `docs/20-phase7-design.md` D7-2:
  * the reference `radii` table has only N/O/C/H/S/P, and `output_pdb_as_xyzrn` skips any atom whose
    type is absent — so **F/Cl/Br/I are silently dropped from the surface**. Harmless for proteins,
    unacceptable for drug-like ligands. We emit our own xyzrn with Bondi radii for halogens etc.
  * the reference derives the element from `atom_name[0]`, which types `CL1` as carbon. We take the
    element from RDKit.

Usage (inside the container):
    python p7_lig_surface.py --sdf <lig.sdf> --out-prefix <dir/id> [--mol2 <lig.mol2>]
Writes `<out_prefix>.ply` (verts/faces/normals + charge/hbond/hphob) and prints a JSON report.
"""
from __future__ import print_function

import argparse
import json
import os
import random
import sys
from subprocess import PIPE, Popen

import numpy as np

from default_config.masif_opts import masif_opts
from default_config.global_vars import msms_bin
from input_output.save_ply import save_ply
from triangulation.computeAPBS import computeAPBS
from triangulation.computeCharges import assignChargesToNewMesh
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.compute_normal import compute_normal
from triangulation.fixmesh import fix_mesh
from triangulation.ligand_charges import computeChargeHelperMol, prepare_rdmol

import pymesh
from rdkit import Chem
from rdkit.Chem import AllChem

LIG_RESNAME = "LIG"
LIG_CHAIN = "L"

# D7-2: the reference table extended so halogens and the other elements that actually occur in
# drug-like ligands get a surface at all. Bondi van der Waals radii.
RADII = {
    "H": 1.20, "C": 1.74, "N": 1.54, "O": 1.40, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "B": 1.92, "SE": 1.90, "SI": 2.10, "AS": 1.85,
}
DEFAULT_RADIUS = 1.80          # metals / anything unlisted


def load_mol(sdf, mol2=None):
    """Bound-pose RDKit mol with explicit hydrogens and PDB atom names."""
    mol = None
    if sdf and os.path.exists(sdf):
        mol = Chem.SDMolSupplier(sdf, sanitize=True, removeHs=False)[0]
        if mol is None:
            mol = Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)[0]
            if mol is not None:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    if mol is None and mol2 and os.path.exists(mol2):
        mol = Chem.MolFromMol2File(mol2, sanitize=True, removeHs=False)
    if mol is None:
        return None, "unreadable"
    if mol.GetNumConformers() == 0:
        return None, "no conformer"
    # MSMS needs the protonated surface, and the hbond geometry needs donor hydrogens.
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    if n_h == 0:
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception as exc:                                    # noqa: BLE001
            return None, "AddHs failed: %s" % exc
    return name_atoms(mol), None


def name_atoms(mol):
    """Give every atom a unique <=4-char PDB name; both ligand channel helpers key on this."""
    counts, info = {}, []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().upper()
        counts[sym] = counts.get(sym, 0) + 1
        nm = ("%s%d" % (sym, counts[sym]))[:4]
        ri = Chem.AtomPDBResidueInfo()
        ri.SetName(nm.ljust(4))
        ri.SetResidueName(LIG_RESNAME)
        ri.SetResidueNumber(1)
        ri.SetChainId(LIG_CHAIN)
        ri.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(ri)
        info.append(nm)
    if len(set(info)) != len(info):
        raise ValueError("atom names not unique")
    return mol


def write_xyzrn(mol, path):
    """x y z radius 1 <chain>_<res>_<ins>_<resname>_<atomname>_<color>, the format MSMS+MaSIF expect."""
    conf = mol.GetConformer()
    n = 0
    with open(path, "w") as fh:
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol().upper()
            r = RADII.get(sym, DEFAULT_RADIUS)
            nm = atom.GetPDBResidueInfo().GetName().strip()
            p = conf.GetAtomPosition(atom.GetIdx())
            color = {"O": "Red", "N": "Blue"}.get(sym, "Green")
            full = "%s_%d_x_%s_%s_%s" % (LIG_CHAIN, 1, LIG_RESNAME, nm, color)
            fh.write("%.06f %.06f %.06f %.6f 1 %s\n" % (p.x, p.y, p.z, r, full))
            n += 1
    return n


def _read_text(path):
    """MSMS writes a banner containing a non-ASCII byte. The reference `read_msms` opens with the
    locale default encoding, so it works under a UTF-8 locale and dies under POSIX/C — an
    environment-dependent failure that would hit a fraction of a 5,000-job array. Decode explicitly."""
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", "replace").rstrip().split("\n")


def read_msms_local(file_root):
    """Same semantics as `input_output/read_msms.read_msms`, without the locale dependency."""
    md = _read_text(file_root + ".vert")
    nv = int(md[2].split()[0])
    vertices = np.zeros((nv, 3))
    normals = np.zeros((nv, 3))
    names = [""] * nv
    for i in range(3, 3 + nv):
        f = md[i].split()
        vertices[i - 3] = [float(f[0]), float(f[1]), float(f[2])]
        normals[i - 3] = [float(f[3]), float(f[4]), float(f[5])]
        names[i - 3] = f[9]
    md = _read_text(file_root + ".face")
    nf = int(md[2].split()[0])
    faces = np.zeros((nf, 3), dtype=int)
    for i in range(3, 3 + nf):
        f = md[i].split()
        faces[i - 3] = [int(f[0]) - 1, int(f[1]) - 1, int(f[2]) - 1]
    return vertices, faces, normals, names


def run_msms(xyzrn_base):
    args = [msms_bin, "-density", "3.0", "-hdensity", "3.0", "-probe", "1.5",
            "-if", xyzrn_base + ".xyzrn", "-of", xyzrn_base, "-af", xyzrn_base]
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if not os.path.exists(xyzrn_base + ".vert"):
        raise RuntimeError("MSMS produced no vertices: %s" % err.decode("utf-8", "ignore")[-400:])
    return read_msms_local(xyzrn_base)


def write_mol2(mol, path):
    """mol2 for pdb2pqr --ligand. Gasteiger charges; OpenBabel via a PDB round-trip if available."""
    try:
        from openbabel import openbabel
        pdb_block = Chem.MolToPDBBlock(mol, flavor=4)
        conv = openbabel.OBConversion()
        conv.SetInAndOutFormats("pdb", "mol2")
        obmol = openbabel.OBMol()
        conv.ReadString(obmol, pdb_block)
        conv.WriteFile(obmol, path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:                                               # noqa: BLE001
        return False


def normalize_electrostatics(elec, lo=-3.0, hi=3.0):
    """Verbatim `read_data_from_surface.normalize_electrostatics` (clip, then map to [-1,1])."""
    e = np.clip(np.copy(elec), lo, hi)
    return 2.0 * ((e - lo) / (hi - lo)) - 1.0


def shape_index(mesh):
    """Per-vertex shape index from discrete curvature — the same five lines the reference uses.

    Also returns how many vertices needed the `H^2 - K < 0` clamp, which is the honest measure of
    how well the discrete estimator copes with a small, highly curved mesh (Phase-7 risk R2)."""
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    n_clamped = int((elem < 0).sum())
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    denom = k1 - k2
    denom[np.abs(denom) < 1e-12] = 1e-12
    si = np.arctan((k1 + k2) / denom) * (2 / np.pi)
    return np.nan_to_num(si, nan=0.0, posinf=0.0, neginf=0.0), n_clamped


def vertex_owner(vertices, mol):
    """Nearest HEAVY atom index per vertex — the ligand analogue of `surf.vertex_surf_idx`."""
    conf = mol.GetConformer()
    heavy, coords = [], []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            p = conf.GetAtomPosition(a.GetIdx())
            heavy.append(a.GetIdx())
            coords.append([p.x, p.y, p.z])
    coords = np.array(coords)
    d = ((np.asarray(vertices)[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.int64)


def build(args):
    rep = {"id": args.id or os.path.basename(args.out_prefix)}
    mol, err = load_mol(args.sdf, args.mol2)
    if mol is None:
        rep.update(ok=False, err=err)
        return rep
    rep["n_atom"] = mol.GetNumAtoms()
    rep["n_heavy"] = mol.GetNumHeavyAtoms()

    tmp = os.environ.get("TMPDIR", masif_opts["tmp_dir"])
    base = os.path.join(tmp, "p7lig_%d" % random.randint(1, 10 ** 7))
    rep["n_xyzrn"] = write_xyzrn(mol, base + ".xyzrn")

    verts1, faces1, normals1, names1 = run_msms(base)
    rep["n_msms_vert"] = len(verts1)
    if len(verts1) < args.min_vertices:
        rep.update(ok=False, err="MSMS gave only %d vertices" % len(verts1))
        return rep

    # ---- chemistry channels on the RAW MSMS vertices (reference order) ----
    _m, name_to_idx, donor_h, acceptors = prepare_rdmol(mol)
    hbond = np.zeros(len(verts1))
    missed = 0
    for ix, nm in enumerate(names1):
        atom_name = nm.split("_")[4]
        if atom_name not in name_to_idx:
            missed += 1
            continue
        hbond[ix] = computeChargeHelperMol(mol, name_to_idx[atom_name], donor_h, acceptors,
                                           verts1[ix])
    rep["vertices_unmatched_to_atom"] = missed
    hphob = computeHydrophobicity(names1, ligand_code=LIG_RESNAME, rdmol=mol)

    # ---- mesh regularisation at the SAME resolution the protein uses ----
    mesh = pymesh.form_mesh(np.array(verts1), np.array(faces1))
    regular = fix_mesh(mesh, masif_opts["mesh_res"])
    rep["n_vert"] = int(regular.vertices.shape[0])
    rep["n_face"] = int(regular.faces.shape[0])
    if rep["n_vert"] < args.min_vertices:
        rep.update(ok=False, err="fix_mesh collapsed to %d vertices" % rep["n_vert"])
        return rep
    vnorm = compute_normal(regular.vertices, regular.faces)
    hbond_r = assignChargesToNewMesh(regular.vertices, verts1, hbond, masif_opts)
    hphob_r = assignChargesToNewMesh(regular.vertices, verts1, hphob, masif_opts)

    # ---- electrostatics: same APBS solver as the protein side (D7-5) ----
    lig_pdb = base + ".pdb"
    with open(lig_pdb, "w") as fh:
        fh.write(Chem.MolToPDBBlock(mol, flavor=4))
    charges = np.zeros(len(regular.vertices))
    rep["apbs"] = "none"
    want = args.apbs_mode
    if want in ("pdb2pqr", "both"):
        mol2 = base + ".mol2"
        if write_mol2(mol, mol2):
            try:
                from triangulation.ligand_utils import amide_to_single_bond
                amide_to_single_bond(mol2)      # pdb2pqr rejects the 'am' bond type outright
            except Exception:                                       # noqa: BLE001
                pass
            try:
                q = computeAPBS(regular.vertices, lig_pdb, base, mol2)
                rep["apbs"] = "pdb2pqr_ligand"
                charges = q
            except Exception as exc:                                # noqa: BLE001
                rep["apbs_err_pdb2pqr"] = str(exc)[-200:]
    if want in ("selfpqr", "both") or rep["apbs"] == "none":
        try:
            q = apbs_from_selfpqr(mol, regular.vertices, base)
            if want == "both" and rep["apbs"] == "pdb2pqr_ligand":
                # S1 evidence for D7-5: do the two parameterisations actually disagree?
                a, b = charges, q
                rep["apbs_corr_pdb2pqr_vs_selfpqr"] = float(np.corrcoef(a, b)[0, 1])
                rep["apbs_selfpqr_range"] = [float(b.min()), float(b.max())]
            else:
                charges = q
                rep["apbs"] = "selfpqr"
        except Exception as exc:                                    # noqa: BLE001
            rep["apbs_err_selfpqr"] = str(exc)[-200:]

    # ---- the 4 vertex channels, in EXACTLY the convention the protein side is stored in ----
    # `read_data_from_surface.py` builds them as: si from mesh curvature; charge = APBS/10 (save_ply
    # normalize_charges) then normalize_electrostatics (clip +-3 -> [-1,1]); hbond raw; hphob/4.5.
    # We replicate that here instead of running 04-masif_precompute, whose 12 A geodesic patches are
    # meaningless on a 150-vertex mesh -- the self-row si it would return is this same per-vertex
    # curvature quantity anyway.
    si, n_clamped = shape_index(regular)
    feat = np.stack([si,
                     hbond_r,
                     normalize_electrostatics(charges / 10.0),
                     hphob_r / 4.5], axis=1).astype(np.float32)
    rep["si_clamped_frac"] = round(float(n_clamped) / max(len(si), 1), 4)

    out_ply = args.out_prefix + ".ply"
    d = os.path.dirname(out_ply)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    save_ply(out_ply, regular.vertices, regular.faces, normals=vnorm, charges=charges,
             normalize_charges=True, hbond=hbond_r, hphob=hphob_r)
    np.save(args.out_prefix + "_verts.npy", np.asarray(regular.vertices, dtype=np.float64))
    np.save(args.out_prefix + "_faces.npy", np.asarray(regular.faces, dtype=np.int64))
    np.save(args.out_prefix + "_normals.npy", np.asarray(vnorm, dtype=np.float64))
    np.save(args.out_prefix + "_feat.npy", feat)
    np.save(args.out_prefix + "_vatom.npy", vertex_owner(regular.vertices, mol))
    rep.update(ok=True, ply=out_ply,
               si=[float(si.min()), float(si.max()), float(si.mean())],
               hbond=[float(hbond_r.min()), float(hbond_r.max())],
               hphob=[float(hphob_r.min()), float(hphob_r.max())],
               charge_raw=[float(np.min(charges)), float(np.max(charges))],
               charge_norm=[float(feat[:, 2].min()), float(feat[:, 2].max())])
    for ext in (".xyzrn", ".vert", ".face", ".area", ".pdb", ".mol2"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    return rep


def write_pqr_rows(mol, fh, start_serial=1, resnum=1):
    """Append ligand atoms to a PQR: Gasteiger charge + the D7-2 radius, APBS's native input."""
    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()
    n = 0
    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        q = atom.GetDoubleProp("_GasteigerCharge")
        if not np.isfinite(q):
            q = 0.0
        r = RADII.get(atom.GetSymbol().upper(), DEFAULT_RADIUS)
        nm = atom.GetPDBResidueInfo().GetName().strip()
        # whitespace-delimited, the same dialect pdb2pqr --whitespace emits, so one parser reads
        # both halves of a merged PQR and long coordinates can never run two fields together.
        fh.write("ATOM %d %s %s %s %d %.3f %.3f %.3f %.4f %.4f\n"
                 % (start_serial + i, nm[:4], LIG_RESNAME, LIG_CHAIN, resnum,
                    p.x, p.y, p.z, q, r))
        n += 1
    return n


def run_apbs(pqr_path, vertices, base):
    """Solve PB on an arbitrary PQR and sample the potential at `vertices` (kT/e).

    Shared by the ligand-alone path and the composite path so both sit on the same physical scale;
    the grid is sized off the molecule the way psize.py would."""
    from default_config.global_vars import apbs_bin, multivalue_bin

    # PQR has two dialects: fixed-column, and the whitespace form pdb2pqr --whitespace writes.
    # The trailing five fields are x y z charge radius in both, so index from the end.
    xyz = []
    with open(pqr_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                f = line.split()
                xyz.append([float(f[-5]), float(f[-4]), float(f[-3])])
    xyz = np.array(xyz)
    if len(xyz) == 0:
        raise RuntimeError("empty PQR: %s" % pqr_path)
    lo, hi = xyz.min(0), xyz.max(0)
    span = hi - lo
    cglen = span + 30.0
    fglen = span + 15.0
    cent = (hi + lo) / 2.0
    dime = np.clip((np.ceil(fglen / 0.5) // 32 * 32 + 1).astype(int), 33, 161)
    inp = base + ".in"
    directory = os.path.dirname(base) or "."
    with open(inp, "w") as fh:
        fh.write("read\n    mol pqr %s\nend\nelec\n    mg-auto\n" % os.path.basename(pqr_path))
        fh.write("    dime %d %d %d\n" % tuple(dime))
        fh.write("    cglen %.3f %.3f %.3f\n" % tuple(cglen))
        fh.write("    fglen %.3f %.3f %.3f\n" % tuple(fglen))
        fh.write("    cgcent %.3f %.3f %.3f\n" % tuple(cent))
        fh.write("    fgcent %.3f %.3f %.3f\n" % tuple(cent))
        fh.write("    mol 1\n    lpbe\n    bcfl sdh\n    pdie 2.0000\n    sdie 78.5400\n"
                 "    srfm smol\n    chgm spl2\n    sdens 10.00\n    srad 1.40\n    swin 0.30\n"
                 "    temp 298.15\n    calcenergy no\n    calcforce no\n"
                 "    write pot dx %s\nend\nquit\n" % os.path.basename(base))
    p = Popen([apbs_bin, os.path.basename(inp)], stdout=PIPE, stderr=PIPE, cwd=directory)
    _o, e = p.communicate()
    if not os.path.exists(base + ".dx"):
        raise RuntimeError("APBS failed: %s" % e.decode("utf-8", "ignore")[-300:])
    with open(base + ".csv", "w") as fh:
        for v in vertices:
            fh.write("%f,%f,%f\n" % (v[0], v[1], v[2]))
    p = Popen([multivalue_bin, os.path.basename(base) + ".csv", os.path.basename(base) + ".dx",
               os.path.basename(base) + "_out.csv"], stdout=PIPE, stderr=PIPE, cwd=directory)
    p.communicate()
    charges = np.zeros(len(vertices))
    with open(base + "_out.csv") as fh:
        for ix, line in enumerate(fh):
            if ix < len(charges):
                charges[ix] = float(line.split(",")[3])
    for ext in (".in", ".dx", ".csv", "_out.csv"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    return charges


def apbs_from_selfpqr(mol, vertices, base):
    """D7-5: ligand electrostatics with a self-written PQR, used for the WHOLE ligand corpus.

    `pdb2pqr --ligand` succeeded on only 15/50 sampled ligands even after the amide-bond fix, so
    routing part of the corpus through it would split the charge channel across two scales. Where
    both ran they correlate at 0.927 (S1), so uniformity costs little."""
    pqr = base + ".pqr"
    with open(pqr, "w") as fh:
        write_pqr_rows(mol, fh)
    try:
        return run_apbs(pqr, vertices, base)
    finally:
        try:
            os.remove(pqr)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", default=None, help="batch mode: one PDBbind id per line")
    ap.add_argument("--pdbbind-dir", default="/scratch/ymeng/masif-graph/data/pdbbind")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sdf", default=None)
    ap.add_argument("--mol2", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--min-vertices", type=int, default=30)
    ap.add_argument("--apbs-mode", choices=["selfpqr", "pdb2pqr", "both", "none"],
                    default="selfpqr",
                    help="D7-5: selfpqr is the default so the WHOLE ligand corpus sits on one\n                          charge scale; 'both' additionally reports their correlation (S1).")
    args = ap.parse_args()
    if args.ids_file:
        if not args.out_dir:
            ap.error("--out-dir is required in batch mode")
        if not os.path.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        sys.exit(batch(args))
    if not args.sdf or not args.out_prefix:
        ap.error("single mode needs --sdf and --out-prefix")
    try:
        rep = build(args)
    except Exception as exc:                                        # noqa: BLE001
        rep = {"id": args.id, "ok": False, "err": "%s: %s" % (type(exc).__name__, exc)}
    print(json.dumps(rep))
    sys.exit(0 if rep.get("ok") else 1)


if __name__ == "__main__":
    main()
