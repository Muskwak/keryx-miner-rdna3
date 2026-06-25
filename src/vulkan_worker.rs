//! In-process Vulkan PoW worker (RDNA3). Replaces the CUDA/OpenCL cdylib plugins: a [`Plugin`]
//! that yields a single [`Worker`] driving the verified `keryx-vulkan` kHeavyHash kernel on the
//! GPU. Registered directly into the [`PluginManager`](crate::PluginManager) by `main` — no shared
//! library is loaded. The miner's existing GPU loop drives it through the `Worker` trait unchanged.

use clap::ArgMatches;
use keryx_vulkan::khh::{KhhGpu, MATRIX_LEN};
use log::info;
use rand::{thread_rng, RngCore};

use crate::{Error, Plugin, Worker, WorkerSpec};

/// Nonces per `calculate_hash` dispatch. The kernel grinds the whole batch before returning, so
/// this trades latency for launch-overhead amortisation. Overridable via `KERYX_VULKAN_WORKLOAD`.
const DEFAULT_WORKLOAD: usize = 1 << 20;

fn configured_workload() -> usize {
    std::env::var("KERYX_VULKAN_WORKLOAD").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(DEFAULT_WORKLOAD)
}

/// In-process plugin exposing the single RDNA3 GPU as one PoW worker.
pub struct VulkanPlugin {
    workload: usize,
}

impl Default for VulkanPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanPlugin {
    pub fn new() -> Self {
        Self { workload: configured_workload() }
    }
}

impl Plugin for VulkanPlugin {
    fn name(&self) -> &'static str {
        "vulkan"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn get_worker_specs(&self) -> Vec<Box<dyn WorkerSpec>> {
        vec![Box::new(VulkanWorkerSpec { workload: self.workload })]
    }
    fn process_option(&mut self, _matches: &ArgMatches) -> Result<usize, Error> {
        info!("Vulkan PoW: 1 RDNA3 worker (workload {} nonces/dispatch)", self.workload);
        Ok(1)
    }
}

pub struct VulkanWorkerSpec {
    workload: usize,
}

impl WorkerSpec for VulkanWorkerSpec {
    fn id(&self) -> String {
        "Vulkan RDNA3".to_string()
    }
    fn build(&self) -> Box<dyn Worker> {
        Box::new(VulkanKhhWorker::new(self.workload).expect("Vulkan kHeavyHash worker init failed"))
    }
}

/// kHeavyHash PoW worker backed by the `keryx-vulkan` GPU kernel.
pub struct VulkanKhhWorker {
    gpu: KhhGpu,
    workload: usize,
    nonce_cursor: u64,
    last_winner: u64,
    id: String,
}

impl VulkanKhhWorker {
    fn new(workload: usize) -> Result<Self, Error> {
        let gpu = KhhGpu::new().map_err(|e| -> Error { e.into() })?;
        let id = gpu.device_name().to_string();
        info!("Vulkan kHeavyHash worker on: {}", id);
        Ok(Self { gpu, workload, nonce_cursor: thread_rng().next_u64(), last_winner: 0, id })
    }
}

impl Worker for VulkanKhhWorker {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn load_block_constants(&mut self, hash_header: &[u8; 72], matrix: &[[u16; 64]; 64], target: &[u64; 4]) {
        // 72-byte pow header → 9 little-endian u64 words (pre_pow_hash || time || 0^32).
        let mut header = [0u64; 9];
        for (i, w) in header.iter_mut().enumerate() {
            *w = u64::from_le_bytes(hash_header[i * 8..i * 8 + 8].try_into().unwrap());
        }
        // [[u16;64];64] (4-bit entries) → flat row-major [u32; 4096] for the SSBO.
        let mut mat = [0u32; MATRIX_LEN];
        for (r, row) in matrix.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                mat[r * 64 + c] = v as u32;
            }
        }
        self.gpu.upload_block(&mat, header, *target);
    }

    fn calculate_hash(&mut self, _nonces: Option<&Vec<u64>>, _nonce_mask: u64, _nonce_fixed: u64) {
        // Solo full-block mining: contiguous nonce range from a moving cursor. (Applying
        // nonce_mask/nonce_fixed inside the kernel for pool shares is a later refinement; the
        // primary RDNA3 path is solo, where nonce_mask is all-ones and nonce_fixed is 0.)
        let start = self.nonce_cursor;
        self.last_winner = self.gpu.mine(start, self.workload as u32).unwrap_or(0);
        self.nonce_cursor = self.nonce_cursor.wrapping_add(self.workload as u64);
    }

    fn sync(&self) -> Result<(), Error> {
        Ok(()) // mine() already blocks on the GPU fence
    }

    fn get_workload(&self) -> usize {
        self.workload
    }

    fn copy_output_to(&mut self, nonces: &mut Vec<u64>) -> Result<(), Error> {
        nonces[0] = self.last_winner;
        Ok(())
    }
}
