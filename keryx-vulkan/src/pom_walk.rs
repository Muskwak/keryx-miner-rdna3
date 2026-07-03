//! GPU Proof-of-Model walk: dispatch the `pom_walk` compute shader over a resident weight blob to
//! find the lowest nonce in a batch whose `pom_pow_value <= target`. The folds are byte-identical
//! to `src/pom.rs`, so a nonce found here builds a `PomProof` the node accepts.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

/// SPIR-V for the PoM walk, compiled from `shaders/pom_walk.comp` by build.rs.
const POM_WALK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pom_walk.spv"));

/// POM_WALK_STEPS — must match `pom::POM_WALK_STEPS` and the node.
pub const POM_WALK_STEPS: u32 = 256;

/// Compute `floor(2^64 / d)` — the magic multiplier for Granlund-Montgomery fast 64-bit modulo.
/// The shader uses `mul_hi(x, mod_magic)` to approximate the quotient, then corrects with a
/// conditional subtract. `floor(2^64 / d)` guarantees the quotient never overshoots (unlike
/// `ceil(2^64 / d)`, which can produce q = floor(x/d)+1 for some x, making x − q*d wrap near
/// 2^64 — a single correction branch can't fix it → GPU page fault → DEVICE_LOST).
fn mod_magic(d: u64) -> u64 {
    u64::MAX / d
}

/// Push-constant block — layout MUST match the `Push` block in `pom_walk.comp` (std430: thirteen
/// u64 at 0..104, then three u32 at 104, 108, 112; total 120 bytes incl. tail pad). `mod_magic` is
/// `floor(2^64 / n_chunks)`, precomputed for the fast 64-bit modulo (Granlund-Montgomery mul_hi).
#[repr(C)]
#[derive(Clone, Copy)]
struct PomPush {
    p: [u64; 4],
    t: [u64; 4],
    timestamp: u64,
    n_chunks: u64,
    start_nonce: u64,
    shard_mask: u64, // chunks_per_shard - 1
    mod_magic: u64,  // floor(2^64 / n_chunks) — fast_mod multiplier
    k: u32,
    batch: u32,
    shard_shift: u32, // log2(chunks_per_shard)
}

const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Chunks per shard: 2^25 × 32 B = 1 GiB, comfortably under the AMD 2 GiB `maxMemoryAllocationSize`
/// single-allocation cap, with headroom for driver overhead. Power of two so the shader maps a
/// chunk to (shard, offset) with a shift + mask instead of 64-bit divide.
const SHARD_CHUNKS: u64 = 1 << 25;

/// Max nonces per GPU dispatch. Matching POM_BATCH (1<<20) eliminates the sub-dispatch loop:
/// one dispatch per miner call instead of four, cutting fence-wait + readback overhead. With 256
/// threads per workgroup, 1M nonces = 4096 workgroups, well under the 2 s Windows TDR limit
/// (a single walk dispatch takes ~150 ms at 6+ MH/s).
const MAX_DISPATCH_NONCES: u32 = 1 << 20;

/// Resident GPU PoM miner: the weight blob lives in a storage buffer; `mine` re-dispatches the
/// walk over nonce batches. Build once per mining tier (the weight blob is large).
pub struct PomWalkGpu {
    vk: Vk,
    kernel: Kernel,
    shards: Vec<GpuBuffer>, // weight blob split into ≤1 GiB device-address buffers
    addr_table: GpuBuffer,  // bound SSBO: one u64 GPU address per shard
    winner: GpuBuffer,
    n_chunks: u64,
    shard_chunks: u64,
}

impl PomWalkGpu {
    /// Upload the canonical weight blob (`weight_words` = the model's quant bytes as little-endian
    /// u64 words, `n_chunks * 4` of them) and compile the walk kernel on the RDNA3 GPU.
    pub fn new(weight_words: &[u64], n_chunks: u64) -> Result<Self, String> {
        Self::new_sharded(weight_words, n_chunks, SHARD_CHUNKS)
    }

    /// Like [`new`](Self::new) but with an explicit shard size (chunks per shard, power of two).
    /// Lets tests force a multi-shard layout without multi-GiB allocations.
    pub fn new_sharded(weight_words: &[u64], n_chunks: u64, shard_chunks: u64) -> Result<Self, String> {
        if n_chunks == 0 || weight_words.len() as u64 != n_chunks * 4 {
            return Err(format!(
                "weight blob size mismatch: {} words for {} chunks (expected {})",
                weight_words.len(),
                n_chunks,
                n_chunks * 4
            ));
        }
        if !shard_chunks.is_power_of_two() {
            return Err(format!("shard_chunks must be a power of two, got {shard_chunks}"));
        }
        let vk = Vk::new()?;
        let spirv = ash::util::read_spv(&mut Cursor::new(POM_WALK_SPV)).map_err(|e| e.to_string())?;
        // Two descriptor bindings: the winner buffer and the shard address table. The (large) weight
        // shards are reached by device address, not bound as descriptors.
        let kernel = vk.make_kernel(&spirv, 2, std::mem::size_of::<PomPush>() as u32)?;

        // Split the blob on chunk boundaries into device-address shards; collect their GPU addresses.
        let n_shards = n_chunks.div_ceil(shard_chunks);
        let mut shards: Vec<GpuBuffer> = Vec::with_capacity(n_shards as usize);
        let mut addrs: Vec<u64> = Vec::with_capacity(n_shards as usize);
        for s in 0..n_shards {
            let first_word = (s * shard_chunks * 4) as usize;
            let last_word = (((s + 1) * shard_chunks).min(n_chunks) * 4) as usize;
            let slice = &weight_words[first_word..last_word];
            // Device-local VRAM (staged copy): the walk's random reads are ~100x faster here than
            // host-visible memory — host-visible overran the TDR watchdog → DEVICE_LOST.
            let (buf, addr) = vk.create_device_local_address_buffer(words_as_bytes(slice))?;
            shards.push(buf);
            addrs.push(addr);
        }

        // Address table (tiny — one u64 per shard) bound as a normal storage buffer at binding 1.
        let addr_table = vk.create_buffer((addrs.len() * 8) as u64)?;
        vk.write_buffer(&addr_table, words_as_bytes(&addrs));
        let winner = vk.create_buffer(4)?;

        Ok(Self { vk, kernel, shards, addr_table, winner, n_chunks, shard_chunks })
    }

    /// Name of the GPU the miner is running on.
    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_chunks
    }

    /// Search nonces `[start, start + batch)`. Returns the lowest winning nonce, or None.
    ///
    /// The full batch is dispatched in `MAX_DISPATCH_NONCES`-sized chunks (default 1<<20), well
    /// under the 2 s Windows TDR limit. Sub-batches are ascending, so the first one with any winner
    /// holds the global lowest nonce — returning there is identical to grinding the whole batch,
    /// and skips the rest.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u32) -> Option<u64> {
        let mut done: u32 = 0;
        while done < batch {
            let sub = (batch - done).min(MAX_DISPATCH_NONCES);
            self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
            let push = PomPush {
                p: words4(pre_pow_hash),
                t: words4(target_le),
                timestamp,
                n_chunks: self.n_chunks,
                start_nonce: start + done as u64,
                shard_mask: self.shard_chunks - 1,
                mod_magic: mod_magic(self.n_chunks),
                k: POM_WALK_STEPS,
                batch: sub,
                shard_shift: self.shard_chunks.trailing_zeros(),
            };
            let groups = sub.div_ceil(256); // local_size_x = 256
            self.vk.dispatch(&self.kernel, &[&self.winner, &self.addr_table], push_bytes(&push), groups);

            let mut out = [0u8; 4];
            self.vk.read_buffer(&self.winner, &mut out);
            if let offset @ 0..=0xFFFF_FFFE = u32::from_le_bytes(out) {
                return Some(start + done as u64 + offset as u64);
            }
            done += sub;
        }
        None
    }
}

impl Drop for PomWalkGpu {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.addr_table);
        for shard in &self.shards {
            self.vk.destroy_buffer(shard);
        }
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
