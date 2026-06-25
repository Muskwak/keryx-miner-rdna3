//! GPU Proof-of-Model walk: dispatch the `pom_walk` compute shader over a resident weight blob to
//! find the lowest nonce in a batch whose `pom_pow_value <= target`. The folds are byte-identical
//! to `src/pom.rs`, so a nonce found here builds a `PomProof` the node accepts.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the PoM walk, compiled from `shaders/pom_walk.comp` by build.rs.
const POM_WALK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk.spv"));

/// POM_WALK_STEPS — must match `pom::POM_WALK_STEPS` and the node.
pub const POM_WALK_STEPS: u32 = 256;

/// Push-constant block — layout MUST match the `Push` block in `pom_walk.comp` (scalar std430:
/// eleven u64 at offsets 0..88, then two u32 at 88, 92; total 96 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    k: u32,
    batch: u32,
}

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Resident GPU PoM miner: the weight blob lives in a storage buffer; `mine` re-dispatches the
/// walk over nonce batches. Build once per mining tier (the weight blob is large).
pub struct PomWalkGpu {
    vk: Vk,
    kernel: Kernel,
    weights: GpuBuffer,
    winner: GpuBuffer,
    n_chunks: u64,
}

impl PomWalkGpu {
    /// Upload the canonical weight blob (`weight_words` = the model's quant bytes as little-endian
    /// u64 words, `n_chunks * 4` of them) and compile the walk kernel on the RDNA3 GPU.
    pub fn new(weight_words: &[u64], n_chunks: u64) -> Result<Self, String> {
        if n_chunks == 0 || weight_words.len() as u64 != n_chunks * 4 {
            return Err(format!(
                "weight blob size mismatch: {} words for {} chunks (expected {})",
                weight_words.len(),
                n_chunks,
                n_chunks * 4
            ));
        }
        let vk = Vk::new()?;
        let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_SPV)).map_err(|e| e.to_string())?;
        let kernel = vk.make_kernel(&spirv, 2, std::mem::size_of::<PomPush>() as u32)?;

        let weights = vk.create_buffer(weight_words.len() as u64 * 8)?;
        vk.write_buffer(&weights, words_as_bytes(weight_words));
        let winner = vk.create_buffer(4)?;

        Ok(Self { vk, kernel, weights, winner, n_chunks })
    }

    /// Name of the GPU the miner is running on.
    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    /// Search nonces `[start, start + batch)`. Returns the lowest winning nonce, or None.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        if batch == 0 {
            return None;
        }
        self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
        let push = PomPush {
            p: words4(pre_pow_hash),
            t: words4(target_le),
            timestamp,
            n_chunks: self.n_chunks,
            start_nonce: start,
            k: POM_WALK_STEPS,
            batch,
        };
        let groups = batch.div_ceil(64); // local_size_x = 64
        self.vk.dispatch(&self.kernel, &[&self.weights, &self.winner], push_bytes(&push), groups);

        let mut out = [0u8; 4];
        self.vk.read_buffer(&self.winner, &mut out);
        match u32::from_le_bytes(out) {
            NO_WINNER => None,
            offset => Some(start + offset as u64),
        }
    }
}

impl Drop for PomWalkGpu {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.weights);
        self.vk.destroy_kernel(&self.kernel);
    }
}

/// 32 LE bytes → 4 u64 words (matches `pom::pph_words` / `words4`).
pub fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

fn words_as_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, std::mem::size_of_val(words)) }
}

fn push_bytes(p: &PomPush) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const PomPush as *const u8, std::mem::size_of::<PomPush>()) }
}
