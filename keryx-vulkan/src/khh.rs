//! GPU kHeavyHash (legacy PoW): dispatch the `khh` compute shader over a nonce batch to find the
//! lowest nonce whose final hash `<= target`. The matrix is generated host-side (salt-versioned,
//! full-rank) and uploaded per block; the shader does PowHash → matmul → wave_mix → HeavyHash.

use crate::{GpuBuffer, Kernel, Vk};
use std::io::Cursor;

const KHH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/khh.spv"));

/// 64x64 matrix of 4-bit entries.
pub const MATRIX_LEN: usize = 64 * 64;
const NO_WINNER: u32 = 0xFFFF_FFFF;

/// Push-constant block — must match the `Push` block in `khh.comp` (header[9]+target[4]+start are
/// u64 at 0..112, batch u32 at 112; std430 rounds the block to 120 bytes via the trailing pad).
#[repr(C)]
#[derive(Clone, Copy)]
struct KhhPush {
    header: [u64; 9],
    target: [u64; 4],
    start_nonce: u64,
    batch: u32,
    _pad: u32,
}

/// Resident GPU kHeavyHash miner. Upload a block's matrix/header/target once, then `mine` nonce
/// batches against it.
pub struct KhhGpu {
    vk: Vk,
    kernel: Kernel,
    matrix: GpuBuffer,
    winner: GpuBuffer,
    header: [u64; 9],
    target: [u64; 4],
}

impl KhhGpu {
    pub fn new() -> Result<Self, String> {
        let vk = Vk::new()?;
        let spirv = ash::util::read_spv(&mut Cursor::new(KHH_SPV)).map_err(|e| e.to_string())?;
        let kernel = vk.make_kernel(&spirv, 2, std::mem::size_of::<KhhPush>() as u32)?;
        let matrix = vk.create_buffer((MATRIX_LEN * 4) as u64)?;
        let winner = vk.create_buffer(4)?;
        Ok(Self { vk, kernel, matrix, winner, header: [0; 9], target: [0; 4] })
    }

    pub fn device_name(&self) -> &str {
        self.vk.device_name()
    }

    /// Load a block's constants: the 64x64 4-bit matrix (row-major), the 72-byte pow header as 9
    /// LE u64 words, and the 256-bit target as 4 LE u64 words (`word[3]` most significant).
    pub fn upload_block(&mut self, matrix: &[u32; MATRIX_LEN], header: [u64; 9], target: [u64; 4]) {
        self.vk.write_buffer(&self.matrix, u32s_as_bytes(matrix));
        self.header = header;
        self.target = target;
    }

    /// Search nonces `[start, start + batch)`. Returns the lowest winning nonce, or None.
    pub fn mine(&self, start: u64, batch: u32) -> Option<u64> {
        if batch == 0 {
            return None;
        }
        self.vk.write_buffer(&self.winner, &NO_WINNER.to_le_bytes());
        let push = KhhPush { header: self.header, target: self.target, start_nonce: start, batch, _pad: 0 };
        let groups = batch.div_ceil(64);
        self.vk.dispatch(&self.kernel, &[&self.matrix, &self.winner], push_bytes(&push), groups);
        let mut out = [0u8; 4];
        self.vk.read_buffer(&self.winner, &mut out);
        match u32::from_le_bytes(out) {
            NO_WINNER => None,
            offset => Some(start + offset as u64),
        }
    }
}

impl Drop for KhhGpu {
    fn drop(&mut self) {
        self.vk.destroy_buffer(&self.winner);
        self.vk.destroy_buffer(&self.matrix);
        self.vk.destroy_kernel(&self.kernel);
    }
}

fn u32s_as_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn push_bytes(p: &KhhPush) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const KhhPush as *const u8, std::mem::size_of::<KhhPush>()) }
}
