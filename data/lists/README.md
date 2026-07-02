# Reused PDB dataset lists

These are the **only artifact reused verbatim** from the reference MaSIF-neosurf project.
Copied from `../masif-neosurf-af2/masif/data/masif_ppi_search/lists/` on 2026-07-02.

Each line is a PPI pair id `PDBID_chainA_chainB` (chains may be multi-letter, e.g. `1A14_HL_N`).

| file | lines | use |
|---|---|---|
| `training.txt` | 4943 | training complexes |
| `testing.txt` | 959 | held-out complexes (Phase-1 probe draws from here) |
| `full_list.txt` | 5902 | union / superset |
| `training_seed_benchmark.txt` | 4919 | seed-benchmark split |
| `testing_seed_benchmark.txt` | 983 | seed-benchmark split |

We reuse the **identity of the complexes** only. All surfaces, atoms, graphs, and
descriptors are (re)generated from raw PDBs per `docs/01-phase1-design.md` (D10).
