### Axis 1 — do-no-harm PPI gate (287-clean, dense `pos`)

| model | seeds | HH top5 | HH medR | AA top5 | AA medR | holo→AA drop |
|---|---|---|---|---|---|---|
| random-init (chance) | 1 | 0.015 (1 seed) | 258 (1 seed) | 0.009 (1 seed) | 253 (1 seed) | +0.006 (1 seed) |
| PPI-only (do-no-harm control) | 2 | 0.644 ± 0.007 | 1 ± 0 | 0.654 ± 0.006 | 1 ± 0 | -0.010 ± +0.001 |
| ligand-only (transfer control) | 1 | 0.011 (1 seed) | 262 (1 seed) | 0.013 (1 seed) | 258 (1 seed) | -0.002 (1 seed) |
| Phase-6C combined (ligand = atoms only) | 2 | 0.627 ± 0.018 | 1 ± 0 | 0.639 ± 0.017 | 1 ± 0 | -0.012 ± +0.001 |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | 0.475 ± 0.020 | 9 ± 3 | 0.493 ± 0.029 | 6 ± 4 | -0.019 ± +0.009 |

Frozen MaSIF on the same patches: HH top5 0.084 (medR 110), AA 0.061. n=269, DB=538, chance top5 0.0093.

### Axis 2a — **TRAIN-set** retrieval (the capacity gate: can it fit the ligand axis at all?)

| model | seeds | PPI top5 | P–L top5 | P–L MRR | P–L medR |
|---|---|---|---|---|---|
| random-init (chance) | 1 | 0.025 (1 seed) | 0.013 (1 seed) | 0.023 (1 seed) | 134 (1 seed) |
| PPI-only (do-no-harm control) | 2 | 0.447 ± 0.018 | 0.026 ± 0.003 | 0.028 ± 0.001 | 134 ± 3 |
| ligand-only (transfer control) | 1 | 0.028 (1 seed) | 0.041 (1 seed) | 0.037 (1 seed) | 105 (1 seed) |
| Phase-6C combined (ligand = atoms only) | 2 | 0.448 ± 0.019 | 0.119 ± 0.024 | 0.092 ± 0.018 | 38 ± 7 |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | 0.363 ± 0.010 | 0.111 ± 0.007 | 0.095 ± 0.001 | 32 ± 1 |

- PPI: DB 198, chance top5 0.0254, chance medR ~99
- P–L: DB 281, chance top5 0.0179, chance medR ~140

### Axis 2b — mixed held-out retrieval

| model | seeds | PPI top5 | P–L top5 | P–L MRR | P–L medR |
|---|---|---|---|---|---|
| random-init (chance) | 1 | 0.030 (1 seed) | 0.014 (1 seed) | 0.019 (1 seed) | 150 (1 seed) |
| PPI-only (do-no-harm control) | 2 | 0.595 ± 0.006 | 0.019 ± 0.002 | 0.023 ± 0.002 | 148 ± 0 |
| ligand-only (transfer control) | 1 | 0.028 (1 seed) | 0.036 (1 seed) | 0.036 (1 seed) | 116 (1 seed) |
| Phase-6C combined (ligand = atoms only) | 2 | 0.579 ± 0.003 | 0.054 ± 0.015 | 0.049 ± 0.007 | 74 ± 2 |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | 0.448 ± 0.027 | 0.084 ± 0.012 | 0.066 ± 0.004 | 57 ± 1 |

- PPI: DB 198, chance top5 0.0254, chance medR ~99
- P–L: DB 292, chance top5 0.0172, chance medR ~146

### Axis 2c — held-out, scaffold-unseen subset (clean on protein cluster AND scaffold)

| model | seeds | PPI top5 | P–L top5 | P–L MRR | P–L medR |
|---|---|---|---|---|---|
| random-init (chance) | 1 | - | 0.010 (1 seed) | 0.022 (1 seed) | 100 (1 seed) |
| PPI-only (do-no-harm control) | 2 | - | 0.026 ± 0.003 | 0.032 ± 0.000 | 96 ± 3 |
| ligand-only (transfer control) | 1 | - | 0.042 (1 seed) | 0.041 (1 seed) | 77 (1 seed) |
| Phase-6C combined (ligand = atoms only) | 2 | - | 0.076 ± 0.013 | 0.061 ± 0.010 | 52 ± 2 |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | - | 0.111 ± 0.007 | 0.084 ± 0.003 | 40 ± 0 |

- P–L: DB 192, chance top5 0.0262, chance medR ~96

### Axis 3 — neosurface benchmark (28 cases; median rank, lower is better)

| model | seeds | sep-surf medR | sep-surf no-lig medR | **composite medR** | composite no-lig medR | sep-surf top5 | composite top5 |
|---|---|---|---|---|---|---|---|
| random-init (chance) | 1 | 305 (1 seed) | 260 (1 seed) | 280 (1 seed) | 344 (1 seed) | 0.000 (1 seed) | 0.000 (1 seed) |
| PPI-only (do-no-harm control) | 2 | 286 ± 50 | 271 ± 23 | - | - | 0.000 ± 0.000 | - |
| ligand-only (transfer control) | 1 | 253 (1 seed) | 258 (1 seed) | - | - | 0.000 (1 seed) | - |
| Phase-6C combined (ligand = atoms only) | 2 | 260 ± 6 | 255 ± 42 | - | - | 0.036 ± 0.000 | - |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | 347 ± 15 | 338 ± 16 | 337 ± 2 | 335 ± 13 | 0.018 ± 0.018 | 0.036 ± 0.036 |

DB = 596 chains (568 held-out decoys); chance medR ~298, chance top5 0.0084. `composite` = protein and drug on ONE surface; `composite_noligand` drops the drug's own rows, isolating the drug *reshaping the protein surface*.

### Axis 4 (north star) — ligand-axis holo→AF3-apo robustness

| model | seeds | P(holo)→lig top5 | P(AF3)→lig top5 | drop | holo medR | AF3 medR |
|---|---|---|---|---|---|---|
| random-init (chance) | 1 | 0.014 (1 seed) | 0.021 (1 seed) | -0.007 (1 seed) | 146 (1 seed) | 152 (1 seed) |
| PPI-only (do-no-harm control) | 2 | 0.027 ± 0.005 | 0.011 ± 0.000 | +0.016 ± +0.005 | 139 ± 2 | 144 ± 2 |
| ligand-only (transfer control) | 1 | 0.049 (1 seed) | 0.025 (1 seed) | +0.025 (1 seed) | 90 (1 seed) | 100 (1 seed) |
| Phase-6C combined (ligand = atoms only) | 2 | 0.053 ± 0.021 | 0.046 ± 0.004 | +0.007 ± +0.018 | 72 ± 4 | 86 ± 2 |
| **Phase-7 combined (ligand = atoms + SURFACE)** | 2 | 0.074 ± 0.007 | 0.060 ± 0.000 | +0.014 ± +0.007 | 48 ± 0 | 62 ± 2 |

