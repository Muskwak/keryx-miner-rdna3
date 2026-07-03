//! Proof-of-Model GPU mining — **Vulkan** backend (RDNA3). Replaces the old CUDA `pom_mine` PTX
//! path. Streams the mining tier's GGUF quant bytes into per-GPU Vulkan storage buffers using the
//! canonical name-sorted, 32-byte-chunk layout (identical to [`crate::pom::WeightIndex`]), then
//! drives the verified `keryx_vulkan` PoM walk kernel to find a winning nonce. The host
//! `WeightIndex` is also built so a winning nonce can be turned into a `PomProof`
//! (`State::generate_block_if_pom`).
//!
//! Zero-dup load path: the VRAM blob is filled straight from the GGUF on disk through the
//! `WeightIndex` chunk table (bounded 256 MiB staging window) — the packed full-model host `Vec`
//! the old loader built (~4.6 GiB for the 8B tier, ~25-40 GiB for 32B/70B) no longer exists.
//! Multi-GPU: one resident blob per mining device, keyed by the raw Vulkan device index; every
//! device mines the same tier over the one shared host index.
//!
//! The seed/walk/pow folds are byte-identical across the GPU kernel, `pom.rs`, and the node, so a
//! nonce found here builds a proof the node accepts.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use keryx_vulkan::pom_walk::PomWalkGpu;
use log::info;

/// Lowercase hex of a 32-byte digest, for diagnostics.
fn hex32(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Resident GPU PoM miners, one per mining device (raw Vulkan device index). An entry is dropped
/// to free that device's VRAM (inference has priority on the inference device). `Arc` so `mine`
/// can dispatch without holding the map lock — N GPUs must not serialize each other's batches.
static MINERS: Mutex<Option<HashMap<u32, Arc<PomWalkGpu>>>> = Mutex::new(None);

/// Mining-tier identity for (re)builds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// Number of in-flight one-time index/blob loads (workers intentionally paused, not stalled).
static LOADING: AtomicUsize = AtomicUsize::new(0);

/// Record the mining tier so the miner can build its index + GPU weight blob on first PoM activation.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Whether a PoM index/blob load is in progress on any device (worker intentionally paused).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed) > 0
}

/// Whether the GPU PoM miner is resident and ready on `device`.
pub fn is_installed(device: u32) -> bool {
    MINERS.lock().map(|g| g.as_ref().is_some_and(|m| m.contains_key(&device))).unwrap_or(false)
}

/// The resident miner for `device`, if installed.
fn miner_on(device: u32) -> Option<Arc<PomWalkGpu>> {
    MINERS.lock().ok()?.as_ref()?.get(&device).cloned()
}

/// VRAM bytes occupied by the resident PoM weight blob (`n_chunks * 32`) on the INFERENCE device,
/// or 0 if not installed there. Inference uses this to decide whether the blob can stay resident
/// alongside the served model instead of being unloaded + later re-staged on every challenge.
/// Blobs on other mining devices never compete with inference, so they are not counted.
pub fn resident_blob_bytes() -> u64 {
    miner_on(keryx_vulkan::inference_device_index() as u32).map(|m| m.n_chunks() * 32).unwrap_or(0)
}

/// Drop the INFERENCE device's GPU PoM miner, freeing its weight-blob VRAM so inference (priority)
/// can use that GPU. Mining rebuilds the blob when it next runs there. Blobs on other mining
/// devices stay resident — they do not contend with the served model. The host `WeightIndex`
/// stays (cheap, disk-backed).
pub fn uninstall() {
    let infer = keryx_vulkan::inference_device_index() as u32;
    if let Ok(mut g) = MINERS.lock() {
        if let Some(m) = g.as_mut() {
            m.remove(&infer);
        }
    }
}

/// Search nonces `[start, start + batch)` on `device`. None if not installed or no winner.
pub fn mine(
    device: u32,
    pre_pow_hash: &[u8; 32],
    timestamp: u64,
    target_le: &[u8; 32],
    start: u64,
    batch: u64,
) -> Option<u64> {
    // Clone the Arc out and dispatch lock-free: each device has exactly one worker thread, and
    // holding the map lock across a multi-ms walk batch would serialize the other GPUs.
    let m = miner_on(device)?;
    let batch = batch.min(u32::MAX as u64) as u32;
    m.mine(pre_pow_hash, timestamp, target_le, start, batch)
}

/// Ensure the GPU PoM miner is installed on `device`; build the host possession index (first
/// activation, shared across devices) and stream the weight blob into that device's VRAM if
/// needed. Returns true when ready to mine.
pub fn ensure_installed(daa: u64, device: u32) -> bool {
    if is_installed(device) {
        return true;
    }
    LOADING.fetch_add(1, Ordering::Relaxed);
    let ok = ensure_installed_inner(daa, device);
    LOADING.fetch_sub(1, Ordering::Relaxed);
    ok
}

/// PoM tier index of the mining model at a given block DAA. Recomputed per block (not frozen at
/// index-build time) so the tier reindexing at the very-light hardfork (H2) is applied at the
/// exact boundary — e.g. Gemma 0→1 — rather than from a stale build-time value. The proof's `tier`
/// field MUST come from here, keyed on the block's own DAA, or a post-H2 block carries the stale
/// 4-tier index and the node rejects it (`BadWeightPath`).
pub fn current_tier(daa: u64) -> Option<u8> {
    let (model_id, _) = MINING_TIER.get()?;
    crate::models::pom_tier_index(model_id, daa)
}

fn ensure_installed_inner(daa: u64, device: u32) -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };

    // Build the host possession index once (heavy: hashes every chunk to a disk Merkle tree).
    // Needed to construct the PoM proof for a winning nonce, and it doubles as the zero-dup
    // GPU upload source (its chunk table maps canonical chunks to GGUF file offsets).
    if crate::pom::active_index().is_none() {
        // Build-time tier is used only for logging + the get_or_build_index/set_index bookkeeping;
        // the tier EMITTED in each proof is recomputed per block via `current_tier(daa)`. `daa` here
        // is the block DAA at first activation (>= POM_ACTIVATION_DAA, guaranteed by the caller).
        let tier = match crate::models::pom_tier_index(model_id, daa) {
            Some(t) => t,
            None => return false,
        };

        // Defer the heavy index build until the mining-tier GGUF is fully downloaded. The `.ok`
        // sentinel sits next to model.gguf (written by slm after a verified download). Building from
        // a partial GGUF fails with a confusing partial-read/ENOENT; returning false here just lets
        // the mining loop retry on its next tick once the download lands. Checked via the GGUF's own
        // directory rather than slm's SUPPORTED_SPECS, which holds the legacy (v1) lineup until the
        // post-fork swap and would not list this v2 mining model.
        let model_ready = std::path::Path::new(gguf)
            .parent()
            .map(|d| d.join(".ok").exists())
            .unwrap_or(false);
        if !model_ready {
            info!("PoM: mining-tier model not fully downloaded yet (.ok absent) — deferring index build.");
            return false;
        }

        // Serialize the one-time host index build across PoM workers. Harmless for a single worker,
        // but required once >1 worker exists (multi-GPU): get_or_build_index makes exactly one
        // build. The closure also enforces the consensus-pinned (R_T, N) for this tier — a
        // wrong-quant / corrupt / truncated GGUF is rejected HERE, once, instead of silently
        // producing PoM blocks every one of which the node rejects with BadWeightPath.
        let gguf_path = gguf.clone();
        let expected = crate::models::pinned_pom_anchor(model_id);
        if !crate::pom::get_or_build_index(tier, move || {
            let idx = crate::pom::WeightIndex::build_from_gguf(&gguf_path)?;
            if let Some(anchor) = expected {
                if idx.n_chunks != anchor.chunks {
                    return Err(candle_core::Error::Msg(format!(
                        "PoM: index chunk count {} != consensus-pinned {} — wrong/corrupt GGUF for this tier; \
                         refusing to mine (every block would be rejected)",
                        idx.n_chunks, anchor.chunks
                    )));
                }
                if idx.r_t != anchor.root {
                    return Err(candle_core::Error::Msg(format!(
                        "PoM: computed R_T {} != consensus-pinned root for this tier — wrong/corrupt GGUF; \
                         refusing to mine (every block would be rejected)",
                        hex32(&idx.r_t)
                    )));
                }
                info!("PoM: index R_T + N match the consensus-pinned anchor for tier {}.", tier);
            } else {
                log::warn!("PoM: no consensus-pinned anchor for this model_id — skipping the R_T/N check.");
            }
            Ok(idx)
        }) {
            return false;
        }
    }

    // Stream the canonical weight blob from the GGUF straight into this device's VRAM through the
    // index's chunk table — no packed host copy (the old `load_weight_words` Vec was ~1x model
    // size). The blob N equals the index N by construction (same table), so the proof-vs-blob
    // N-guard the packed loader needed is structural here.
    let (idx, _) = match crate::pom::active_index() {
        Some(x) => x,
        None => return false,
    };
    info!("PoM(vulkan): streaming weight blob into VRAM on device {}…", device);
    let mut source = |first_chunk: u64, out: &mut [u8]| {
        idx.read_chunk_range(first_chunk, out).map_err(|e| format!("GGUF chunk stream failed: {e}"))
    };
    match PomWalkGpu::new_streamed(Some(device as usize), idx.n_chunks, &mut source) {
        Ok(gpu) => {
            info!(
                "PoM(vulkan): GPU miner ready on {} (device {}) — N={} chunks resident",
                gpu.device_name(),
                device,
                idx.n_chunks
            );
            if let Ok(mut g) = MINERS.lock() {
                g.get_or_insert_with(HashMap::new).insert(device, Arc::new(gpu));
            }
            true
        }
        Err(e) => {
            log::error!("PoM(vulkan): GPU miner init failed on device {}: {}", device, e);
            false
        }
    }
}
