DGX Server – Specs & Practical Constraints
Hardware

System: NVIDIA DGX (older generation)

GPU:

1× NVIDIA Tesla V100-SXM2

32 GB VRAM

CPU RAM: ~512 GB system RAM

Storage: Local disk available (used for model caches and checkpoints)

Architecture: x86_64

Network: Internet access available but prefers offline / pre-downloaded models

Software Stack

OS: Linux (DGX OS / Ubuntu-based)

CUDA: CUDA 12.x compatible

PyTorch:

Running PyTorch 2.1–2.3 (CUDA enabled)

Containers:

Uses Docker

Base images often from nvcr.io/nvidia/pytorch

Python: 3.10

UI / Serving:

Gradio + FastAPI

Dynamic ports (multiple instances simultaneously)

Model & Workflow Preferences

Offline-first setup

Models stored locally and mounted into containers

Shared cache:

/root/.cache/huggingface


mounted into containers:

-v /root/.cache/huggingface:/root/.cache/huggingface


Rebuild image when code changes (repo baked into image)

Multiple concurrent container instances, each:

On a different port

Ideally bound to idle GPUs only

Heavy models: Stable Diffusion 1.5, ControlNet, BrushNet, etc.

Known Limitations / Gotchas

Single GPU only

No model parallelism across GPUs

Large models must fit in 32 GB VRAM

Transformer Engine & Flash-Attn

Must be disabled / uninstalled

Cause symbol mismatch errors with DGX PyTorch builds:

undefined symbol: _ZN3c104impl3cow11cow_deleterEPv


LLaVA / Vision-Language Models

Fragile due to dependency mismatches

Often run in optional / disabled mode

Gradio quirks

Sliders may return floats → must cast to int

Queue callbacks sensitive to type mismatches

SHMEM

Docker default /dev/shm too small

Recommended flags:

--ipc=host --ulimit memlock=-1 --ulimit stack=67108864

What This DGX Is Best At

✅ Stable Diffusion / ControlNet / inpainting
✅ Long-running GPU jobs
✅ Multi-container inference (one GPU per job)
✅ Offline, reproducible pipelines
❌ Not ideal for latest bleeding-edge CUDA extensions
❌ Not ideal for huge LLMs (>32 GB VRAM)