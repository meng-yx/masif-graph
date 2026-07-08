"""Write PyMOL-inspectable examples of the Phase-4 positive sets (dense vs sc-filtered).

For each complex, in the shared holo (interaction) frame:
  <pdb>_L.pdb / <pdb>_R.pdb          the two interacting chains (L=p1/first, R=p2/second)
  <pdb>_dense_L.vert / _dense_R.vert dense-contact positive surface-atom coords (L side / R side)
  <pdb>_sc_L.vert   / _sc_R.vert      sc-filtered positive surface-atom coords
  <pdb>.pml                           loads all of the above + a self-contained `loaddots` command

A `.vert` file is plain text, one point per line `x y z` (float Å), `#`-comment lines ignored.

The dots are the coordinates of the **surface atoms that are members of a positive PAIR** — i.e. the
exact entities the descriptor-separation AUC scores. dense = all touching-surface contacts
(vertex_contacts, cutoff 1.0 Å, no sc gate). sc = the same, additionally gated to MaSIF's
shape-complementarity band sc∈(0.5,1.0). sc is a subset of dense.

Run (masif-graph env, Jed):
  python scripts/p4_viz_positives.py --ids 1A0G_A_B 1A0H_E_D 1A22_A_B 1A2A_C_D --out viz_positives
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from masif_graph.io.reference import load_complex, PDB_DIR
from masif_graph.surface.atoms import build_surface_atoms
from masif_graph.pairs.construct import vertex_contacts, atom_positives_from_vertex_contacts


PML_TEMPLATE = '''# PyMOL visualization of Phase-4 interface positives for {cid}
# L = chain {c1} (p1), R = chain {c2} (p2), in the holo interaction frame.
# Dots = coordinates of surface ATOMS that are members of a positive contact pair.
#   dense (blue/salmon, small)  = all touching-surface contacts (<1.0 A vertex contact)
#   sc    (green/hotpink, big)  = dense + MaSIF shape-complementarity gate sc in (0.5,1.0)  [subset]
# dense: L={n_dense_l} R={n_dense_r} atoms | sc: L={n_sc_l} R={n_sc_r} atoms

# --- self-contained custom commands ---
#   loaddots  <file.vert>  name, color, radius        : draw each `x y z` line as a CGO sphere
#   loadpairs <file_pairs> name, color, radius         : draw a CGO cylinder between each positive
#                                                        PAIR (`xL yL zL  xR yR zR`) -> the L<->R
#                                                        correspondence the dot clouds can't show
python
from pymol import cmd
from pymol.cgo import COLOR, SPHERE, CYLINDER
def _rgb(color):
    return [float(x) for x in (color if not isinstance(color,str) else cmd.get_color_tuple(color))]
def loaddots(fname, name=None, color=(1.0,1.0,1.0), radius=0.5):
    r,g,b = _rgb(color)
    obj=[COLOR, r,g,b]; n=0
    for line in open(fname):
        line=line.strip()
        if not line or line.startswith('#'): continue
        x,y,z=[float(v) for v in line.split()[:3]]
        obj += [SPHERE, x,y,z, float(radius)]; n+=1
    name = name or fname.split('/')[-1].replace('.vert','')
    cmd.load_cgo(obj, name); print(f'loaddots {{fname}} -> {{name}} ({{n}} dots)')
def loadpairs(fname, name=None, color=(1.0,1.0,0.0), radius=0.1):
    r,g,b = _rgb(color); rad=float(radius)
    obj=[]; n=0
    for line in open(fname):
        line=line.strip()
        if not line or line.startswith('#'): continue
        v=[float(t) for t in line.split()[:6]]
        obj += [CYLINDER, v[0],v[1],v[2], v[3],v[4],v[5], rad, r,g,b, r,g,b]; n+=1
    name = name or fname.split('/')[-1].replace('.txt','')
    cmd.load_cgo(obj, name); print(f'loadpairs {{fname}} -> {{name}} ({{n}} pair-lines)')
cmd.extend('loaddots', loaddots)
cmd.extend('loadpairs', loadpairs)
python end

load {cid}_L.pdb, L_prot
load {cid}_R.pdb, R_prot
remove hydro
hide everything
show cartoon, L_prot or R_prot
show lines
show sticks, not polymer
set cartoon_side_chain_helper, on
# color by chain (carbons); hetero atoms keep element (CPK) colors
util.cbc

loaddots {cid}_dense_L.vert, dense_L, marine, 0.35
loaddots {cid}_dense_R.vert, dense_R, salmon, 0.35
loaddots {cid}_sc_L.vert, sc_L, green, 0.6
loaddots {cid}_sc_R.vert, sc_R, hotpink, 0.6

# correspondence connectors: one cylinder per positive PAIR (dense=grey thin, sc=yellow thick)
loadpairs {cid}_dense_pairs.txt, dense_pairs, grey60, 0.06
loadpairs {cid}_sc_pairs.txt, sc_pairs, yellow, 0.14

set two_sided_lighting, on
bg_color grey15
zoom (dense_L or dense_R), 8
'''


def _write_vert(path, coords, header):
    with open(path, "w") as fh:
        fh.write(f"# {header}\n")
        for x, y, z in coords:
            fh.write(f"{x:.3f} {y:.3f} {z:.3f}\n")


def _write_pairs(path, coordL, coordR, pos, header):
    """One line per positive pair: `xL yL zL  xR yR zR` (L endpoint then R endpoint)."""
    with open(path, "w") as fh:
        fh.write(f"# {header}\n")
        for i, j in pos:
            a = coordL[int(i)]; b = coordR[int(j)]
            fh.write(f"{a[0]:.3f} {a[1]:.3f} {a[2]:.3f}  {b[0]:.3f} {b[1]:.3f} {b[2]:.3f}\n")


def process(cid, out_dir):
    pdb_id, c1, c2 = cid.split("_")
    p1, p2 = load_complex(cid)
    s1 = build_surface_atoms(p1.verts, p1.atom_coords, p1.atom_element, p1.atom_resid,
                             p1.desc_straight, p1.desc_flipped, ops=("mean",))
    s2 = build_surface_atoms(p2.verts, p2.atom_coords, p2.atom_element, p2.atom_resid,
                             p2.desc_straight, p2.desc_flipped, ops=("mean",))

    # dense = all touching-surface contacts; sc = + shape-complementarity gate
    vp_dense, _ = vertex_contacts(p1.verts, p2.verts, pos_cutoff=1.0, sc1=None)
    pos_dense = atom_positives_from_vertex_contacts(vp_dense, s1.vertex_surf_idx, s2.vertex_surf_idx)
    vp_sc, _ = vertex_contacts(p1.verts, p2.verts, pos_cutoff=1.0, sc1=p1.sc, sc_band=(0.5, 1.0))
    pos_sc = atom_positives_from_vertex_contacts(vp_sc, s1.vertex_surf_idx, s2.vertex_surf_idx)

    def _lr_coords(pos):
        if len(pos) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        li = np.unique(pos[:, 0]); ri = np.unique(pos[:, 1])
        return s1.coord[li], s2.coord[ri]

    d_l, d_r = _lr_coords(pos_dense)
    sc_l, sc_r = _lr_coords(pos_sc)

    # chain PDBs in the interaction frame (copy the exact reference chain PDBs the coords come from)
    shutil.copyfile(os.path.join(PDB_DIR, f"{pdb_id}_{c1}.pdb"), os.path.join(out_dir, f"{cid}_L.pdb"))
    shutil.copyfile(os.path.join(PDB_DIR, f"{pdb_id}_{c2}.pdb"), os.path.join(out_dir, f"{cid}_R.pdb"))

    _write_vert(os.path.join(out_dir, f"{cid}_dense_L.vert"), d_l, f"{cid} dense positives, L(chain {c1}) surface atoms")
    _write_vert(os.path.join(out_dir, f"{cid}_dense_R.vert"), d_r, f"{cid} dense positives, R(chain {c2}) surface atoms")
    _write_vert(os.path.join(out_dir, f"{cid}_sc_L.vert"), sc_l, f"{cid} sc-filtered positives, L(chain {c1}) surface atoms")
    _write_vert(os.path.join(out_dir, f"{cid}_sc_R.vert"), sc_r, f"{cid} sc-filtered positives, R(chain {c2}) surface atoms")

    # PAIR files: one line per positive PAIR = `xL yL zL  xR yR zR` (preserves the L<->R correspondence
    # that the deduplicated .vert dot clouds lose). Loaded by the `loadpairs` command as connectors.
    _write_pairs(os.path.join(out_dir, f"{cid}_dense_pairs.txt"), s1.coord, s2.coord, pos_dense,
                 f"{cid} dense positive PAIRS: xL yL zL  xR yR zR (L=chain {c1}, R=chain {c2})")
    _write_pairs(os.path.join(out_dir, f"{cid}_sc_pairs.txt"), s1.coord, s2.coord, pos_sc,
                 f"{cid} sc-filtered positive PAIRS: xL yL zL  xR yR zR (L=chain {c1}, R=chain {c2})")

    with open(os.path.join(out_dir, f"{cid}.pml"), "w") as fh:
        fh.write(PML_TEMPLATE.format(cid=cid, c1=c1, c2=c2,
                                     n_dense_l=len(d_l), n_dense_r=len(d_r),
                                     n_sc_l=len(sc_l), n_sc_r=len(sc_r)))
    return dict(cid=cid, dense_pairs=len(pos_dense), sc_pairs=len(pos_sc),
                dense_L=len(d_l), dense_R=len(d_r), sc_L=len(sc_l), sc_R=len(sc_r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", default="viz_positives")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for cid in args.ids:
        try:
            info = process(cid, args.out)
            print(f"{cid}: dense pairs={info['dense_pairs']} (L{info['dense_L']}/R{info['dense_R']}) "
                  f"| sc pairs={info['sc_pairs']} (L{info['sc_L']}/R{info['sc_R']})", flush=True)
        except Exception as e:
            print(f"{cid}: FAIL {type(e).__name__}: {e}", flush=True)
    print(f"\nwrote examples -> {args.out}/  (open a <id>.pml in PyMOL: `pymol {args.out}/<id>.pml`)")


if __name__ == "__main__":
    main()
