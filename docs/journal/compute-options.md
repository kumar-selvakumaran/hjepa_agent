# Compute options for training

Written 2026-11-15. Labels: see [README.md](README.md). Prices come from blog comparisons, not provider pages, as of about Sep to Oct 2026: **[Unverified]**, confirm on live pages before paying. Nothing here was measured; see the probe plan at the end.

## 1. Alignment

- **[Kumar]** Target: the full training for the use case finishes in **1 to 2 hours wall-clock**, whatever compute that takes.
- **[Agreed]** Defaults: the 1 to 2 hours covers training only (data staged beforehand); the target is one full training run, not all experiments.
- **[Kumar]** Local machine: i7-1355U (10 cores, 12 threads), 31.7 GB RAM, RTX 3050 laptop with 4 GB, 207 GB free on C and 184 GB free on D.
- **[Kumar]** Hoped-for local time: about 4 to 5 hours on that GPU. **[Open]** unmeasured.

## 2. What the reference papers trained on

| Run | Transitions | Frame | Data (raw uint8, my arithmetic) | Model | Weights (fp32) | GPU, memory | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LeWM Two-Room | about 0.9M | 224x224x3, 150.5 KB | about 139 GB | about 15M | 60 MB | not given | "a few hours" |
| LeWM Push-T | about 3.9M | 150.5 KB | about 590 GB | about 15M | 60 MB | not given | same claim |
| LeWM Cube, Reacher | about 2M each | 150.5 KB | about 301 GB each | about 15M | 60 MB | not given | same claim |
| PLDM Two-Rooms | 3M | 2x64x64, 8.2 KB | about 25 GB | 2.2M | 8.8 MB | V100, 16 or 32 GB | cap of 1 day per experiment |
| HWM Diverse Maze | 5M | 64x64x3, 12.3 KB | about 61 GB | about 53k | 0.2 MB | not reported | not reported |
| H-JEPA FourRoom | 1.0M | 150.5 KB | about 151 GB | not stated | not stated | H200, 141 GB | not reported |
| H-JEPA AntMaze (3 levels) | 10.05M | 12.3 KB | about 124 GB | not stated | not stated | H200, 141 GB | not reported |

- LeWM, H-JEPA: 5 to 10 epochs (LeWM 10; H-JEPA 5 to 9). PLDM paper: 500 to 2000 GPU days for the whole research effort on V100s.
- H-JEPA's SLURM template asks for 1 GPU, 16 CPUs, 120 GB RAM and a 48 h limit per run (a ceiling, not a runtime).
- No paper gives wall-clock training time for these runs, and none names a small GPU.
- Ours: 64x48x3 = 9.2 KB per frame, so 1M transitions is about 9 GB and 4M about 37 GB raw.
- A 64x48 image is not automatically cheap with a ViT: patch 4 gives 192 tokens, similar to LeWM's 224px at patch 14 (256 tokens); patch 8 gives 48.

## 3. Machine needed (estimates, **[Unverified]**)

| | Small (1M, 9 GB) | Large (4M, 37 GB) |
| --- | --- | --- |
| GPU memory | 16 to 24 GB | 16 to 24 GB |
| RAM | 32 GB | 64 GB, or read from disk |
| CPU | 8 vCPU | 8 to 16 vCPU |
| Disk | 50 GB | 100 GB |

The local machine meets all but the GPU.

## 4. Rental options and costs

| Option | Price | Notes |
| --- | --- | --- |
| Kaggle free | $0 | About 30 h/week of one P100 or two T4s, 12 h sessions; quota approximate |
| Colab Pro | $9.99/mo, 100 units | Pro+ $49.99/mo, 500 or 600 units (sources disagree); units per GPU-hour not retrieved |
| Vast.ai RTX 4090 (24 GB) | $0.34 to 0.50/h | Cheapest, third-party hosts, can be interrupted |
| RunPod RTX 4090 | $0.34/h community, $0.69 to 0.74/h secure | Listing: 24 GB GPU, 6 vCPU, 41 GB RAM at $0.69/h |
| RunPod A100 80 GB | about $1.39/h | |
| RunPod H100 | $2.89 (PCIe) to $3.99 (SXM)/h | |
| Lambda A100 / H100 | $1.79 to 1.99 / about $3.99 per h | 8x H100 about $32/h |
| AWS g6.xlarge (L4, 24 GB) | $0.8048/h | RAM probably 16 GiB: **[Unverified]**, too small |
| AWS g5.xlarge (A10G, 24 GB) | $1.006/h | |
| AWS p5.48xlarge (8x H100) | about $55/h | Overkill |

Storage: RunPod network volume $0.07/GB/month (100 GB = $7/month); AWS EBS gp3 $0.08/GB/month; S3 $0.023/GB/month; AWS egress free for the first 100 GB/month. Instances bill while running, storage by the month: a deleted-after-run 100 GB volume costs about $0.05 per 5 h.

## 5. Sizing to the 1 to 2 hour target

GPUs needed about = job size in 4090-hours / (target hours x 0.8 scaling efficiency).

| Job size (4090-hours) | GPUs for 2 h | GPUs for 1 h | Cost at $0.69/h | Cost at $0.34 to 0.50/h |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | about $0.70 | $0.34 to 0.50 |
| 5 | 4 | 7 | about $4.30 | $2.10 to 3.10 |
| 20 | 13 | 25 | about $17 | $8.50 to 12.50 |

- **[Proposed]** Job size is unknown until measured. If the hoped 4 to 5 h on the 3050 holds, a 4090 might do it in about 0.5 to 1 h (speed ratio about 5 to 10x is a guess).
- **[Proposed]** Prefer one big GPU (A100 or H100) before multi-GPU DDP; parallelise sweeps as separate single-GPU pods (no scaling loss).
- **[Proposed]** Fixed overheads (data upload, environment install) can eat a 1 to 2 hour budget: 37 GB at 100 Mbps is about 50 minutes; stage the data on a network volume or generate it on the rented machine.
- **[Proposed]** Budget for a first month of experiments: about $15 to 25.

## 6. Plan

1. Throughput probe on the local 4 GB card: `scratch/probe_throughput.py`, output `scratch/probe_throughput.txt` (step time, peak memory, hours per epoch for LeWM-like models at 64x48, patch 4 and 8).
2. Same probe for about 10 minutes on a rented A100 or 4090 to measure the real speed ratio.
3. Set the GPU count from the measured job size using section 5.
