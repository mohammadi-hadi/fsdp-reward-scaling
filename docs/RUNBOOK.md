# Renting a box and running the sweep

Roughly eight GPU-hours of work. Everything that can fail on a laptop should have failed on a
laptop before any of this starts.

## Before renting

```bash
make lint test smoke && make table && git diff --exit-code && git push
```

If that is not clean, the box is just an expensive place to debug.

## Provider notes

* **Lambda** — a plain Ubuntu box over SSH with CUDA already there. Simplest mental model.
  4x L40S availability comes and goes, and stopping an instance loses it.
* **RunPod** — cheapest per GPU-hour, start from a PyTorch container template so torch is
  already installed, and use a network volume so a checkpoint outlives the pod. Community-tier
  pods can be preempted, which for this repository is either a hazard or a free entry for
  `docs/FAILURES.md`.
* **Modal** — no SSH box at all: declare `gpu="L40S:4"` and run torchrun inside, billed per
  second. Excellent for re-running the sweep unattended, poor for the first interactive hour.

Do the first session on Lambda or RunPod. Move the sweep to Modal only to repeat it.

## Hardware

4x L40S 48GB, single node, Ubuntu 22.04 with CUDA 12.x, at least 200 GB of disk (a 4B
checkpoint is about 64 GB and you want two), and 128 GB of RAM.

48 GB is not arbitrary. On 24 GB the 1.7B single-GPU baseline does not fit either, so there is
no baseline to compute scaling efficiency against. On 40 GB the 1.7B baseline fits at about
30 GiB but the 4B two-way shard needs roughly 36 GiB, which is too close to the edge to plan
around. 48 GB gives a real 1-GPU baseline *and* a real OOM, and both are results.

## The sequence

```bash
git clone https://github.com/mohammadi-hadi/fsdp-reward-scaling && cd fsdp-reward-scaling
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,train]"
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"

# This one line explains most of the scaling numbers. Save it.
nvidia-smi topo -m | tee runs/_env-topo.txt

huggingface-cli download Qwen/Qwen3-1.7B-Base    # start the downloads first
huggingface-cli download Qwen/Qwen3-4B-Base      # and read topo.txt while they run

bash scripts/smoke_cpu.sh                        # gloo, seconds, cheapest failures first
bash scripts/run_2gpu.sh train.steps=5           # then the CUDA path, still cheap

bash scripts/bench_sweep.sh                      # the sweep, about two hours
bash scripts/failure_drill.sh                    # about one hour
make table && git add runs/ docs/FAILURES.md && git commit && git push
```

Checkpoints stay on the box. Export the reward model and push it to the Hub separately; there
is no reason to move 64 GB of optimizer state.

## Cost

About 6-7 hours of work, so budget 8. At 2026 on-demand rates of roughly $0.80-1.10 per
L40S-hour a 4x node is $3.2-4.4/hour, so **$26-35, call it $30-60 with fumbling and one
re-run**. A 4x A100-40GB node is nearer $5/hour, so about $42 for the same eight hours.
**Those rates are estimates and were not checked at the time of writing; check at booking.**

The dominant hidden cost is idle time while models download, which is why the downloads start
before anything else.

## If the budget is tight, cut in this order

1. The 4B rows except 4-GPU full-shard and the 1-GPU OOM. The OOM costs under a minute and
   carries the whole argument.
2. The weak-scaling table. Keep strong scaling.
3. The HSDP row and the activation-checkpointing-off row.
4. Shorten the real training run and state the token count plainly.

**Never cut** the two reference runs, the failure drill, or the resharded resume. Every
portfolio repository has a throughput table. Almost none has a measured resume tolerance and a
real SIGKILL.

## Zenodo

The toggle is web-only, so it is a manual step, and the order matters: flip the toggle on
zenodo.org, confirm the webhook appears on the GitHub repository, and only then cut the tag.
Zenodo ignores releases published before the toggle was on.
