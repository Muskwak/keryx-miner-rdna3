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
| **PoM** (possession walk) | Vulkan compute shader | walk over the model weights resident in **device-local VRAM**, sharded across ≤1 GiB buffers reached by buffer-device-address (an 8B+ model's blob exceeds AMD's 4 GiB SSBO range and 2 GiB single-allocation limits). Bit-exact vs `pom::walk_final`, verified on a 7900 XT. |
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
| *(default)* | Dolphin-Llama3-8B | 8 GB | ✅ |
| `--high` | Qwen3-32B (Q4_K_M) | 24 GB | tight / no |
| `--very-high` | Llama-3.3-70B | 48 GB | no |

> Post-hardfork (OPoI v2 / PoM), **1 GPU = 1 tier**: each tier proves possession of and serves
> exactly the single model above — the cumulative "serve everything below my tier" behaviour is
> dropped, because a PoM GPU is bound to the one model whose weights are resident in VRAM.

The miner is **GPU-only by default** (no CPU mining threads); pass `--threads N` (`-t`) to add CPU
PoW workers if you want them.

### Solo vs pool

`--keryxd-address` takes either a `grpc://` node (solo) or a `stratum+tcp://` pool URL. For pools:

```bash
./keryx-miner --mining-address keryx:YOUR_ADDRESS \
  --keryxd-address stratum+tcp://krx.suprnova.cc:4401 \
  --worker rig1 \
  --password d=1000
```

- `--worker` is sent as `address.worker` so the pool credits shares per rig.
- `--password` is the stratum `mining.authorize` password; on suprnova-style pools it requests a
  fixed difficulty (e.g. `d=1000`). Default `x` = the pool's own (vardiff) difficulty.

### All options

```bash
./keryx-miner --help
```

### Useful environment variables

| Var | Meaning |
|-----|---------|
| `KERYX_LLAMA_SERVER` | full path to the `llama-server` (Vulkan) binary |
| `KERYX_VULKAN_WORKLOAD` | nonces per PoW dispatch (default `1048576`) |
| `KERYX_POW_ONLY` | `1` = mine kHeavyHash shares only; skip OPoI models + `llama-server` (no PoM) |
| `KERYX_SKIP_LEGACY_MODELS` | `1` = don't download the pre-fork lineup; only the post-fork (PoM) model is fetched |
| `KERYX_POM_KEEP_RESIDENT` | `1` = keep the PoM weight blob resident across inference when VRAM fits (skips reload) |
| `GLSLC` / `VULKAN_SDK` | (build only) locate `glslc` for shader compilation |

---

## Status & limitations

- ✅ **Verified on a 7900 XT:** both compute kernels are bit-exact against the host/`keccak`
  references; the full miner builds clean and the integrated Vulkan probe detects the GPU.
- ✅ **Live PoM mining** has been exercised end-to-end against the post-hardfork mainnet
  (keryx-node **v1.2.8**, OPoI v2 / PoM active at DAA `37,780,000`): the Dolphin-8B tier mines at
  **~17–18 MH/s** on a 7900 XT and the pool **accepts** the submitted PoM proofs.
- ✅ **PoM weight blob is device-local (VRAM).** It is staged into ≤1 GiB device-local shards
  reached by buffer-device-address — required because an 8B model's ~4.6 GiB blob exceeds AMD's
  4 GiB `maxStorageBufferRange` and 2 GiB `maxMemoryAllocationSize`. Random reads hit GDDR6, not
  PCIe. Each GPU dispatch is also bounded to avoid the Windows TDR watchdog.
- ✅ **Pool shares** work: the Vulkan kHeavyHash kernel applies `nonce_mask`/`nonce_fixed` for the
  pool's extranonce sub-range, and the stratum client handles short-notify DAA inheritance so PoM
  stays active on Short-only pools post-fork.
- ✅ **VRAM capability gate runs on AMD:** total VRAM is queried via Vulkan (the largest
  device-local heap), not `nvidia-smi`, so the model-vs-VRAM filter announces only the tiers your
  card can actually serve (a 7900 XT comfortably runs `--light` and the default tier).
- ⚠️ **Pool version gate:** some pools (suprnova) reject post-fork PoM shares from miners that don't
  advertise `keryx-miner-supr/0.6.3+` in `mining.subscribe`; this fork advertises a compatible
  identity so its (valid) proofs are accepted.
- ⚠️ **Known benign:** an occasional panic in `MinerManager`'s shutdown/reconnect path (a worker
  thread exits before the drop-time join) — harmless under a supervised restart loop; a clean-up
  candidate.

---

## Connect

* **Website:** [keryx-labs.com](https://keryx-labs.com)
* **X (Twitter):** [@Keryx_Labs](https://x.com/Keryx_Labs)
* **Discord:** [Join the Community](https://discord.gg/U9eDmBUKTF)

---

## Dev Fund

**Disabled in this fork** — `devfund_percent` is forced to `0`, so no blocks are diverted and
**100% of mining rewards go to the miner**. (The upstream `--devfund-percent XX.YY` flag still
parses but is overridden.)
