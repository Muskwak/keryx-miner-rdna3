# Keryx Miner — RDNA3 / Vulkan fork

A fork of [keryx-miner](https://github.com/Keryx-Labs/keryx-miner) that runs **entirely on AMD
RDNA3 GPUs (e.g. Radeon RX 7900 XT/XTX) via Vulkan** — no CUDA, no OpenCL, no CPU compute. It
combines GPU PoW (kHeavyHash), GPU Proof-of-Model possession mining (PoM), and on-chain AI
inference (OPoI — Optimistic Proof of Inference).

> Upstream is NVIDIA/CUDA-only (the engine is candle + CUDA, and the build needs `nvcc`). This fork
> replaces every GPU compute path with Vulkan and runs the OPoI models through a prebuilt
> **llama.cpp Vulkan** server, so the miner builds and runs on an AMD-only host.

---

## What runs where

| Workload | Backend | Notes |
|---|---|---|
| **PoW** (kHeavyHash) | Vulkan compute shader | `keccak-f1600 → 64×64 matmul → wave_mix → keccak`. Bit-exact vs the host reference, verified on a 7900 XT. |
| **PoM** (possession walk) | Vulkan compute shader | walk over the model weights resident in a GPU storage buffer. Bit-exact vs `pom::walk_final`, verified on a 7900 XT. |
| **OPoI inference** (Dolphin-8B / Qwen3-32B / Gemma / Llama) | prebuilt **llama.cpp Vulkan** `llama-server` | launched as a child process with all layers offloaded (`-ngl 999`); the miner talks to it over localhost HTTP. |
| **OPoI fraud-proof commitment** | CPU (integer) | `model_fixed::forward` — a deterministic, bit-exact 32-byte fold required by consensus. Not a GPU/LLM workload; unchanged. |

The custom kernels live in the [`keryx-vulkan`](keryx-vulkan/) crate (uses [`ash`](https://crates.io/crates/ash);
shaders compiled GLSL→SPIR-V with `glslc`). Both kernels have bit-exactness tests that run on the
real GPU (`cargo test -p keryx-vulkan`).

---

## Requirements

**Build:**
- Rust + Cargo ([rustup.rs](https://rustup.rs/))
- `protoc` (protobuf compiler)
- **Vulkan SDK** — provides `glslc`, used at build time to compile the compute shaders to SPIR-V.
  Set `VULKAN_SDK` (the installer does this) or put `glslc` on `PATH`, or point `GLSLC` at it.

**Run:**
- An AMD RDNA3 GPU + recent driver (the Vulkan runtime loader `vulkan-1` ships with the AMD
  Adrenalin / Mesa RADV driver — no Vulkan SDK needed at runtime).
- A **prebuilt llama.cpp `llama-server` (Vulkan build)** — see [Inference setup](#inference-setup).

---

## Build from source

```bash
git clone <this-fork> keryx-miner-rdna3
cd keryx-miner-rdna3
cargo build --release
```

Binary: `target/release/keryx-miner` (`.exe` on Windows). No `nvcc`, no CUDA toolkit, no model
SDKs are required.

---

## Inference setup

OPoI inference is mandatory, and this fork serves it through `llama-server` (Vulkan). Download the
prebuilt release for your OS from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) — the **Vulkan** asset, e.g.
`llama-b####-bin-win-vulkan-x64.zip` on Windows — and make `llama-server` discoverable in one of:

1. `<miner_dir>/llama/llama-server[.exe]`  (next to the miner binary), **or**
2. `<miner_dir>/llama-server[.exe]`, **or**
3. on `PATH`, **or**
4. point `KERYX_LLAMA_SERVER` at the full path.

The miner launches it automatically with full GPU offload and serves the active model tier. The
GGUF model files are downloaded on demand over IPFS on first run (same as upstream).

---

## Usage

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS
```

### Inference tiers (OPoI)

| Flag | Models | Min VRAM | Fits a 7900 XT (20 GB)? |
|------|--------|----------|--------------------------|
| `--light` | Gemma-3-4B | 4 GB | ✅ |
| *(default)* | Gemma-3-4B + Dolphin-8B | 8 GB | ✅ |
| `--high` | + Qwen3-32B (Q4_K_M) | 24 GB | tight / no |
| `--very-high` | Llama-3.3-70B | 48 GB | no |

The miner is **GPU-only by default** (no CPU mining threads); pass `--mining-threads N` to add CPU
PoW workers if you want them.

### All options

```bash
./keryx-miner --help
```

### Useful environment variables

| Var | Meaning |
|-----|---------|
| `KERYX_LLAMA_SERVER` | full path to the `llama-server` (Vulkan) binary |
| `KERYX_VULKAN_WORKLOAD` | nonces per PoW dispatch (default `1048576`) |
| `GLSLC` / `VULKAN_SDK` | (build only) locate `glslc` for shader compilation |

---

## Status & limitations

- ✅ **Verified on a 7900 XT:** both compute kernels are bit-exact against the host/`keccak`
  references; the full miner builds clean and the integrated Vulkan probe detects the GPU.
- ⏳ **Live end-to-end mining** (against a real Keryx node, with `llama-server` + downloaded models)
  has not been exercised in this environment — bring your own node/models to validate.
- **PoM weight blob** is currently uploaded to a host-visible buffer. That is correct but reads
  over PCIe; a device-local (VRAM) staging upload is the next performance step for the PoM tier.
- **Pool shares:** the Vulkan PoW worker mines a contiguous nonce range (solo full-block). Applying
  `nonce_mask`/`nonce_fixed` inside the kernel for pool shares is a follow-up; solo PoM/PoW is the
  primary RDNA3 path.
- **VRAM capability gate:** the upstream model-vs-VRAM filter uses `nvidia-smi` and no-ops on AMD,
  so announce only tiers your card can actually serve (a 7900 XT comfortably runs `--light` and the
  default tier).

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

## Dev Fund

2% of mining rewards support development by default.

```bash
--devfund-percent XX.YY
```
