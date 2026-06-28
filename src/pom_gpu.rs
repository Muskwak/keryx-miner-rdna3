//! Proof-of-Model GPU mining — **Vulkan** backend (RDNA3). Replaces the old CUDA `pom_mine` PTX
//! path. Loads the mining tier's GGUF quant bytes into a Vulkan storage buffer using the canonical
//! name-sorted, 32-byte-chunk layout (identical to [`crate::pom::WeightIndex`]), then drives the
//! verified `keryx_vulkan` PoM walk kernel to find a winning nonce. The host `WeightIndex` is also
//! built so a winning nonce can be turned into a `PomProof` (`State::generate_block_if_pom`).
//!
//! The seed/walk/pow folds are byte-identical across the GPU kernel, `pom.rs`, and the node, so a
//! nonce found here builds a proof the node accepts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_core::quantized::gguf_file;
use candle_core::Device;
use keryx_vulkan::pom_walk::PomWalkGpu;
use log::info;

/// The resident GPU PoM miner. `Option` so it can be dropped to free VRAM (inference has priority).
static MINER: Mutex<Option<PomWalkGpu>> = Mutex::new(None);

/// Mining-tier identity for (re)builds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// True while the heavy one-time index/blob load runs (worker intentionally paused, not stalled).
static LOADING: AtomicBool = AtomicBool::new(false);

/// Record the mining tier so the miner can build its index + GPU weight blob on first PoM activation.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Whether a PoM index/blob load is in progress (worker intentionally paused).
pub fn is_loading() -> bool {
    LOADING.load(Ordering::Relaxed)
}

/// Whether the GPU PoM miner is resident and ready.
pub fn is_installed() -> bool {
    MINER.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// VRAM bytes occupied by the resident PoM weight blob (`n_chunks * 32`), or 0 if not installed.
/// Inference uses this to decide whether the blob can stay resident alongside the served model
/// instead of being unloaded + later re-staged on every challenge.
pub fn resident_blob_bytes() -> u64 {
    MINER.lock().ok().and_then(|g| g.as_ref().map(|m| m.n_chunks() * 32)).unwrap_or(0)
}

/// Drop the GPU PoM miner, freeing its weight-blob VRAM so inference (priority) can use the GPU.
/// Mining rebuilds the blob when it next runs. The host `WeightIndex` stays (cheap, disk-backed).
pub fn uninstall() {
    if let Ok(mut g) = MINER.lock() {
        *g = None;
    }
}

/// Search nonces `[start, start + batch)` on the GPU. None if not installed or no winner.
pub fn mine(pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
    let g = MINER.lock().ok()?;
    let m = g.as_ref()?;
    let batch = batch.min(u32::MAX as u64) as u32;
    m.mine(pre_pow_hash, timestamp, target_le, start, batch)
}

/// Ensure the GPU PoM miner is installed; build the host possession index (first activation) and
/// upload the weight blob to VRAM if needed. Returns true when ready to mine.
pub fn ensure_installed() -> bool {
    if is_installed() {
        return true;
    }
    LOADING.store(true, Ordering::Relaxed);
    let ok = ensure_installed_inner();
    LOADING.store(false, Ordering::Relaxed);
    ok
}

fn ensure_installed_inner() -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };

    // Build the host possession index once (heavy: hashes every chunk to a disk Merkle tree).
    // Needed to construct the PoM proof for a winning nonce.
    if crate::pom::active_index().is_none() {
        let tier = match crate::models::pom_tier_index(model_id) {
            Some(t) => t,
            None => return false,
        };
        // Serialize the one-time host index build across PoM workers. Harmless for a single worker,
        // but required once >1 worker exists: build_from_gguf writes a per-process Merkle tree, so
        // concurrent builds would clobber the same file. get_or_build_index makes exactly one build.
        let gguf_path = gguf.clone();
        if !crate::pom::get_or_build_index(tier, || crate::pom::WeightIndex::build_from_gguf(&gguf_path)) {
            return false;
        }
    }

    // Load the canonical weight blob into host RAM, then upload it to the GPU PoM kernel.
    info!("PoM(vulkan): loading weight blob into VRAM…");
    let (words, n_chunks) = match load_weight_words(gguf) {
        Ok(v) => v,
        Err(e) => {
            log::error!("PoM(vulkan): weight blob load failed: {}", e);
            return false;
        }
    };

    // N-guard: the GPU blob's chunk count must equal the host index, else proofs would be rejected.
    if let Some((idx, _)) = crate::pom::active_index() {
        if n_chunks != idx.n_chunks {
            log::error!(
                "PoM(vulkan): blob N={} != index N={} — refusing to mine (would produce rejected blocks)",
                n_chunks, idx.n_chunks
            );
            return false;
        }
    }

    match PomWalkGpu::new(&words, n_chunks) {
        Ok(gpu) => {
            info!("PoM(vulkan): GPU miner ready on {} — N={} chunks resident", gpu.device_name(), n_chunks);
            if let Ok(mut g) = MINER.lock() {
                *g = Some(gpu);
            }
            true
        }
        Err(e) => {
            log::error!("PoM(vulkan): GPU miner init failed: {}", e);
            false
        }
    }
}

/// Read the GGUF's quantized tensors in canonical (name-sorted) order and pack their full 32-byte
/// chunks into little-endian u64 words — the exact layout `pom::WeightIndex` indexes and the node
/// pins in `R_T`. Returns (words, n_chunks) with `words.len() == n_chunks * 4`.
fn load_weight_words(gguf_path: &str) -> candle_core::Result<(Vec<u64>, u64)> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
    let content = gguf_file::Content::read(&mut file)?;
    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort(); // canonical order — must match WeightIndex / the node R_T

    let mut words: Vec<u64> = Vec::new();
    let mut n_chunks: u64 = 0;
    for name in &names {
        let qt = content.tensor(&mut file, name, &device)?;
        let bytes = qt.data()?;
        let full = bytes.len() / 32; // drop any sub-chunk remainder, like WeightIndex
        if full == 0 {
            continue;
        }
        for w in bytes[..full * 32].chunks_exact(8) {
            words.push(u64::from_le_bytes(w.try_into().unwrap()));
        }
        n_chunks += full as u64;
    }
    if n_chunks == 0 {
        return Err(candle_core::Error::Msg("PoM(vulkan): model produced 0 chunks".into()));
    }
    Ok((words, n_chunks))
}
