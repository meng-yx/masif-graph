---
name: connect-to-kuma
description: >-
  How to reach the Kuma GPU cluster from the Jed login node to submit or monitor SLURM
  jobs. Use whenever you need to run anything on GPUs (H100 / L40S), sbatch/squeue/sinfo
  on Kuma, or stage code/data for a Kuma job — Jed and Kuma are SEPARATE SLURM clusters,
  so you cannot sbatch to Kuma directly; you must ssh into Kuma first. Also use if an ssh
  host-key prompt or "REMOTE HOST IDENTIFICATION HAS CHANGED" error appears for Kuma.
---

# Connecting Jed → Kuma

Jed (CPU login node, has internet) and Kuma (GPU cluster) are **two separate SLURM
clusters**. You cannot `sbatch` from Jed to Kuma — `ssh` into Kuma and run SLURM there.
`/home` and `/work` are shared between them; `/scratch` is **not**.

## The working command

```bash
ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch '<command on kuma>'
```

- Key `~/.ssh/id_ed25519` has **no passphrase** → unattended ssh works with **no
  ssh-agent**. `BatchMode=yes` makes it fail fast instead of hanging if auth breaks.
- `~/.ssh` is on shared `/home`, so keys/`authorized_keys`/`known_hosts` are the same file
  on both clusters. If key auth ever regresses, ensure `~/.ssh/id_ed25519.pub` is in
  `~/.ssh/authorized_keys` (one edit covers both clusters).

## Host keys — verify, don't blindly accept

`kuma.hpc.epfl.ch` load-balances two login nodes (`10.91.48.4`, `10.91.48.5`), both
presenting the same keys. If you ever have to re-add Kuma to `known_hosts`, accept **only**
keys matching these fingerprints (current as of 2026-07, cross-checked against
<https://scitas-doc.epfl.ch/supercomputers/kuma/>):

- ED25519 `SHA256:VU3simBjo2CoUePsABLhZ/HpW+anz231EU3rfurZDFo`
- ECDSA   `SHA256:vpM/BzmJapiUU3o6hbm2zlKFN93D8QE3xObVdh8x4hM`
- RSA     `SHA256:u3v9urAmgx03w1xUZR6WOxyXAoDoyTcBbbiYbR4IeMc`

Never disable host-key checking. If the live key doesn't match **and** doesn't match
SCITAS's docs, STOP and ask the human (could be a rotation — or an attack).

## Running SLURM jobs on Kuma

- **Partitions (no default — always pass `-p`):** `h100` (4× H100 94 GB/node, FP64),
  `l40s` (8× L40S 48 GB/node, FP32); plus `mig12gb` / `mig24gb` (H100 MIG slices for
  small jobs). Verify with `ssh … kuma… 'sinfo -s'`.
- **QOS (`-q`):** `normal` (default, ≤8 nodes, 3-day), `long` (7-day), `build`
  (4 h, 0 GPU — compiling), `debug` (1 h, high priority). A 24 h training job fits `normal`.
- **Account/allocation:** `upthomae`, budget cap 10,000 CHF. Every `sbatch` prints a cost
  estimate; check it before large jobs.
- **Staging path (shared, writable):** `/work/upthomae/Meng`. Put code/data there or under
  `/home`. **Never** rely on `/scratch` crossing between clusters.

Example — trivial no-GPU smoke test (proven to work):

```bash
ssh -i ~/.ssh/id_ed25519 ymeng@kuma.hpc.epfl.ch 'sbatch --qos=debug --partition=h100 \
  --gres=gpu:0 --time=00:02:00 --output=/work/upthomae/Meng/smoke-%j.out \
  --wrap="hostname; nvidia-smi -L || echo no-gpu"'
```

## Guardrails
- Do **not** start GPU training or large transfers just to "test" the link — this skill is
  about connectivity and small SLURM ops. Heavy runs need explicit human go-ahead.
- If the Kuma MOTD shows a policy restriction, stop and surface it.

Full setup history, the host-key rotation story, and backups left behind:
`docs/02-jed-to-kuma-connection.md`. When doing autonomous ML work on the cluster, also
follow the `ml-research-guardrails` skill.
