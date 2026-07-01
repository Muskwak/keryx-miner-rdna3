/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored lineup (5 tiers):
///   --very-light  Qwen3-1.7B-abliterated         (Qwen)   — 4 GB+ GPU (PoM tier 0, post-H2)
///   --light       Gemma-3-4B-it-abliterated      (Google) — any GPU (6 GB+)
///   (default)     Dolphin-3.0-Llama-3.1-8B       (Llama)  — RTX 3060 12GB / 3070
///   --high        Qwen3-32B-abliterated (Q4_K_M) (Qwen)   — 24 GB (3090 / 4090 / 5090)
///   --very-high   Llama-3.3-70B-abliterated      (Meta)   — Q4 48 GB (pre-H2) → Q2_K_L 32 GB / 5090 (post-H2)
///
/// All GGUF weights + tokenizers are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Full-precision safetensors (one or more shards).
    Safetensors,
    /// GGUF quantized — LLaMA/LLaMA3 architecture.
    Gguf,
    /// GGUF quantized — Qwen2 architecture (legacy DeepSeek-R1-32B, pre-OPoI-v2 lineup).
    GgufQwen2,
    /// GGUF quantized — Qwen3 architecture (Qwen3-32B).
    GgufQwen3,
    /// GGUF quantized — Gemma 3 architecture (Gemma-3-4B, baseline tier).
    GgufGemma3,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    pub tokenizer_cid: &'static str,
    /// Unused for GGUF (architecture embedded in file).
    pub config_cid: &'static str,
    /// Safetensors: one entry per shard. GGUF: single entry.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + CUDA workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

pub const GEMMA_3_4B: ModelSpec = ModelSpec {
    name: "gemma-3-4b",
    // CIDv0[2..34] of model.gguf — mlabonne/gemma-3-4b-it-abliterated Q4_K_M
    model_id: [
        0xad, 0x50, 0xad, 0x0b, 0xd4, 0x61, 0xd8, 0xab,
        0x44, 0xef, 0xc0, 0x21, 0x49, 0x89, 0xeb, 0x33,
        0x29, 0x16, 0x85, 0xef, 0x4a, 0xde, 0x22, 0xa0,
        0xf4, 0xf2, 0x17, 0xd0, 0x32, 0x66, 0xd8, 0x37,
    ],
    format: ModelFormat::GgufGemma3,
    tokenizer_cid: "QmTh2MsVfAvWp7grN9rvkF9NkMkCW2PhWez2WbNh81KRXD",
    config_cid: "",
    weight_cids: &["Qma1CbFzWTNhy2ReVjDG1GvM5q2Uy4VhqTbnS9c641jUQ6"],
    dir_name: "Gemma-3-4B",
    // Baseline model — never gated. ~4 GB Q4_K_M; runs on any GPU that can mine.
    min_vram_mb: 0,
};

pub const DOLPHIN_LLAMA3_8B: ModelSpec = ModelSpec {
    name: "dolphin-llama3-8b",
    // CIDv0[2..34] of model.gguf — Dolphin3.0-Llama3.1-8B Q4_K_M
    model_id: [
        0x94, 0x21, 0x06, 0x6a, 0x64, 0x00, 0xc9, 0x8b,
        0xa1, 0x37, 0x11, 0x4f, 0x7f, 0x4b, 0x7d, 0x4a,
        0x2d, 0xdf, 0x13, 0xab, 0x16, 0x3a, 0x5d, 0xe3,
        0x8c, 0x01, 0x84, 0x79, 0x3a, 0xf6, 0x31, 0x3a,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmQSe8rZQcTQ6q1xDGquv6s9wpzFT9u27U4wfGVZqwMJgJ",
    config_cid: "",
    weight_cids: &["QmYJtFpaDnVwAVSbzRo42fsb19nLpt8LHe8WVKoyxd4AkZ"],
    dir_name: "Dolphin-Llama3-8B",
    // ~4.9 GB Q4_K_M weights + ~1.6 GB KV/workspace.
    min_vram_mb: 8_000,
};

pub const QWEN3_32B: ModelSpec = ModelSpec {
    name: "qwen3-32b",
    // CIDv0[2..34] of model.gguf — Qwen3-32B-abliterated Q4_K_M (mradermacher)
    model_id: [
        0x65, 0xc6, 0xeb, 0x6f, 0xe1, 0x8b, 0x9e, 0xfd,
        0x80, 0x60, 0xab, 0x9d, 0x2d, 0x03, 0xbb, 0x9b,
        0x01, 0x05, 0x0a, 0x3b, 0x13, 0x78, 0xcb, 0xac,
        0x00, 0x0c, 0x5c, 0xc0, 0xac, 0xdc, 0x0d, 0x2a,
    ],
    format: ModelFormat::GgufQwen3,
    tokenizer_cid: "QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5",
    config_cid: "",
    weight_cids: &["QmVBwp5n3muQJwYNLTHSu3EnzBWviQqfh58FvHvKRfLtam"],
    dir_name: "Qwen3-32B",
    // ~19.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → fits a 24 GB card (3090/4090/5090).
    min_vram_mb: 24_000,
};

pub const LLAMA_3_3_70B: ModelSpec = ModelSpec {
    name: "llama-3.3-70b",
    // CIDv0[2..34] of model.gguf — Llama-3.3-70B-Instruct-abliterated Q4_K_M (bartowski)
    model_id: [
        0x13, 0x29, 0xfb, 0xe2, 0x1b, 0x3f, 0x36, 0xf6,
        0xd0, 0x06, 0x89, 0xfc, 0xaa, 0x74, 0xf7, 0xa2,
        0x22, 0xb8, 0xcc, 0x4c, 0x08, 0xc0, 0x19, 0x1f,
        0xeb, 0x23, 0x97, 0x55, 0xa7, 0x23, 0x42, 0x1e,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmPdTayXcEsfUwMCoMKKcLSv7Dwpp2xVBWELwrG2M7Rhzu"],
    dir_name: "Llama-3.3-70B",
    // ~42.5 GB Q4_K_M weights + ~3.5 GB KV/workspace → 48 GB card (matches the
    // --very-high 46 GB startup gate).
    min_vram_mb: 46_000,
};

pub const QWEN3_1_7B: ModelSpec = ModelSpec {
    name: "qwen3-1.7b",
    // CIDv0[2..34] of model.gguf — mlabonne/Qwen3-1.7B-abliterated Q4_K_M (locally quantized
    // from the mlabonne safetensors; reproducible: re-quantize Q4_K_M and re-add to verify).
    model_id: [
        0x4f, 0x21, 0xdd, 0xeb, 0x7d, 0x62, 0xbd, 0x22,
        0x65, 0xbc, 0x54, 0x23, 0x0d, 0x53, 0x6c, 0xa3,
        0xf1, 0x74, 0x99, 0x27, 0x78, 0x0f, 0x52, 0x8c,
        0x3c, 0x41, 0xfa, 0x29, 0x11, 0xdf, 0x4d, 0x72,
    ],
    format: ModelFormat::GgufQwen3,
    // Qwen3 dense models share the same tokenizer byte-for-byte — identical CID to Qwen3-32B's.
    tokenizer_cid: "QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5",
    config_cid: "",
    weight_cids: &["QmTfYsPusQUPG7t82N7tpwFqXuJZ5yCBGsqP19pDJVanh7"],
    dir_name: "Qwen3-1.7B",
    // ~1.05 GB Q4_K_M — very-light tier, runs comfortably on 4-6 GB GPUs. Never gated.
    min_vram_mb: 0,
};

pub const LLAMA_3_3_70B_Q2: ModelSpec = ModelSpec {
    name: "llama-3.3-70b-q2",
    // CIDv0[2..34] of model.gguf — Llama-3.3-70B-Instruct-abliterated Q2_K_L (bartowski, same
    // base as the Q4_K_M above). Post-H2 very-high tier: fits a 32 GB card (RTX 5090) where the
    // 42.5 GB Q4 never did, so the top tier is finally servable on real consumer hardware.
    model_id: [
        0x6d, 0xf4, 0x6a, 0x78, 0xcb, 0xe4, 0xdc, 0x57,
        0x9f, 0x04, 0xdb, 0xd8, 0x01, 0xf1, 0xa5, 0x20,
        0xb9, 0xea, 0xe2, 0x8c, 0xe7, 0xb5, 0x0c, 0x8d,
        0xa7, 0x87, 0x4b, 0xfa, 0x3f, 0xb5, 0x10, 0x8d,
    ],
    format: ModelFormat::Gguf,
    // Same Llama-3.3 tokenizer as the Q4 (only the quant differs) — already pinned.
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmVjsK1LBMjk24tawUrGyWUEXHQwkcPgeetC5JpNZL7p1J"],
    // Distinct dir from the Q4 70B so both coexist on disk across the H2 transition.
    dir_name: "Llama-3.3-70B-Q2",
    // ~25.5 GiB Q2_K_L weights + ~2.5 GB KV/workspace ≈ 28 GB → needs a 32 GB card (5090).
    // Gate at 30 GB so 24 GB cards are excluded (the Q2 is 5090-exclusive).
    min_vram_mb: 30_000,
};

/// Map a model_id to its Proof-of-Model tier index, matching the node's `POM_TIERS` order.
/// DAA-gated at the very-light hardfork (H2) so the index ordering stays logical (smallest = 0):
///   - daa <  H2 (4-tier): Gemma=0, Dolphin=1, Qwen3-32B=2, Llama-70B-Q4=3.
///   - daa >= H2 (5-tier): Qwen3-1.7B=0, Gemma=1, Dolphin=2, Qwen3-32B=3, Llama-70B-Q2=4.
/// At H2 the top tier's model also changes (70B Q4_K_M → Q2_K_L) so it fits a 32 GB 5090.
/// The gate is mandatory: an archival/IBD node recomputing pre-H2 blocks under the new scheme
/// would assign different tiers → different reward brackets → UTXO divergence. MUST match the
/// node, and the tier must be recomputed per block from that block's DAA (not frozen).
/// Whether `model_id` is one of the Proof-of-Model tier models (any era). DAA-independent —
/// used at startup to pick a mineable PoM model before any block DAA is known (the tier *index*
/// is then computed per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_1_7B.model_id
        || *model_id == GEMMA_3_4B.model_id
        || *model_id == DOLPHIN_LLAMA3_8B.model_id
        || *model_id == QWEN3_32B.model_id
        || *model_id == LLAMA_3_3_70B.model_id
        || *model_id == LLAMA_3_3_70B_Q2.model_id
}


/// Map a model_id to its Proof-of-Model tier index, matching the node's `POM_TIERS` order.
/// DAA-gated at the very-light hardfork (H2) so the index ordering stays logical (smallest = 0):
///   - daa <  H2 (4-tier): Gemma=0, Dolphin=1, Qwen3-32B=2, Llama-70B-Q4=3.
///   - daa >= H2 (5-tier): Qwen3-1.7B=0, Gemma=1, Dolphin=2, Qwen3-32B=3, Llama-70B-Q2=4.
/// At H2 the top tier's model also changes (70B Q4_K_M → Q2_K_L) so it fits a 32 GB 5090.
/// The gate is MANDATORY and MUST match the node's `pom_tiers(very_light_active)`: an
/// archival/IBD node recomputing pre-H2 blocks under the new scheme would assign different tiers
/// → different reward brackets → UTXO divergence. The tier MUST be recomputed per block from that
/// block's own DAA (never frozen at index-build time), or a post-H2 block would carry the stale
/// 4-tier index and be rejected. The `R_T`/`N` consensus check (`pinned_pom_anchor`) is keyed by
/// model_id, so it stays correct across the reorder — but the index emitted in the proof comes
/// from here. None for non-PoM models.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa >= VERY_LIGHT_ACTIVATION_DAA {
        // 5-tier scheme: very-light inserted at 0, the existing tiers shift up by one.
        if *model_id == QWEN3_1_7B.model_id {
            Some(0)
        } else if *model_id == GEMMA_3_4B.model_id {
            Some(1)
        } else if *model_id == DOLPHIN_LLAMA3_8B.model_id {
            Some(2)
        } else if *model_id == QWEN3_32B.model_id {
            Some(3)
        } else if *model_id == LLAMA_3_3_70B_Q2.model_id {
            // Top tier post-H2 = the Q2_K_L (5090-servable), replacing the Q4 (48 GB-only).
            Some(4)
        } else {
            None
        }
    } else {
        // 4-tier scheme (pre-H2): unchanged from OPoI v2.
        if *model_id == GEMMA_3_4B.model_id {
            Some(0)
        } else if *model_id == DOLPHIN_LLAMA3_8B.model_id {
            Some(1)
        } else if *model_id == QWEN3_32B.model_id {
            Some(2)
        } else if *model_id == LLAMA_3_3_70B.model_id {
            Some(3)
        } else {
            None
        }
    }
}

/// Consensus-pinned PoM possession anchor: a model's canonical 32 B-chunk blake3 Merkle root `R_T`
/// and chunk count `N`, produced offline by `pom-rt-builder`.
pub struct PomAnchor {
    pub model_id: [u8; 32],
    pub root: [u8; 32],
    pub chunks: u64,
}

/// Per-model `(R_T, N)` anchors, copied VERBATIM from the node's `POM_TIERS`
/// (`keryx-node consensus/core/src/config/params.rs`). The miner asserts its freshly-built
/// possession index matches the pinned `(root, N)` for the model it mines (see
/// `pom_gpu::ensure_installed_inner`), so a wrong-quant / corrupt / truncated GGUF is caught once
/// at index-build time — instead of silently producing PoM blocks every one of which the node
/// rejects with `BadWeightPath`. Keyed by model_id (not slice position) so the check stays correct
/// even if the node later reorders tiers.
pub const POM_ANCHORS: &[PomAnchor] = &[
    PomAnchor {
        model_id: GEMMA_3_4B.model_id,
        root: [
            0x84, 0x6c, 0xaa, 0x40, 0x0c, 0xf0, 0x14, 0x13, 0x21, 0x18, 0x49, 0x5d, 0x22, 0xe4, 0xbf, 0xa2,
            0x42, 0x45, 0x4e, 0xac, 0x0d, 0x83, 0x5c, 0x3f, 0x8e, 0x63, 0x47, 0xd0, 0x13, 0x9d, 0x1b, 0x7e,
        ],
        chunks: 77_604_776,
    },
    PomAnchor {
        model_id: DOLPHIN_LLAMA3_8B.model_id,
        root: [
            0x13, 0x3f, 0x62, 0x7b, 0x88, 0x2e, 0xf8, 0x56, 0x78, 0x5a, 0x83, 0x98, 0x6a, 0x9b, 0x1a, 0xdf,
            0xed, 0xff, 0xf0, 0x74, 0x4a, 0x1f, 0x94, 0x21, 0xec, 0x4d, 0xa6, 0xe9, 0x46, 0x68, 0x15, 0xde,
        ],
        chunks: 153_528_426,
    },
    PomAnchor {
        model_id: QWEN3_32B.model_id,
        root: [
            0xe2, 0xaa, 0x66, 0x59, 0xaa, 0xb4, 0x38, 0x7e, 0xb5, 0xfd, 0x79, 0x40, 0x9c, 0x0a, 0x1a, 0x68,
            0x86, 0x3a, 0x3d, 0xef, 0x3b, 0x66, 0x2c, 0xb4, 0x06, 0x16, 0x97, 0xf0, 0xea, 0x87, 0xfa, 0x58,
        ],
        chunks: 617_380_448,
    },
    PomAnchor {
        model_id: LLAMA_3_3_70B.model_id,
        root: [
            0x53, 0x5f, 0xc2, 0xac, 0xb6, 0x09, 0x7b, 0x5d, 0xf8, 0x83, 0xec, 0x50, 0x66, 0x9a, 0x7f, 0x48,
            0xdc, 0x9f, 0x3b, 0xd5, 0x98, 0x74, 0x28, 0x59, 0xb8, 0xbb, 0x4c, 0xac, 0x3b, 0x35, 0x26, 0xaa,
        ],
        chunks: 1_328_516_616,
    },
    // ── H2 (post-`very_light_activation`) anchors — copied VERBATIM from the node's `POM_TIERS_H2`
    // (keryx-node consensus/core/src/config/params.rs). Keyed by model_id, so they coexist with the
    // pre-H2 anchors above and `pinned_pom_anchor` resolves the right one regardless of tier order.
    PomAnchor {
        model_id: QWEN3_1_7B.model_id,
        root: [
            0xd0, 0x9a, 0x0b, 0x1c, 0x26, 0x25, 0x69, 0xc2, 0x39, 0xfa, 0xcc, 0xf6, 0x41, 0xf8, 0xe4, 0x35,
            0x4a, 0x15, 0x77, 0x50, 0x1b, 0xa8, 0x42, 0xbc, 0x64, 0x9a, 0x87, 0x6d, 0xe1, 0xaf, 0x9a, 0x5d,
        ],
        chunks: 34_420_544,
    },
    PomAnchor {
        model_id: LLAMA_3_3_70B_Q2.model_id,
        root: [
            0xb9, 0x6c, 0xfc, 0xb5, 0x38, 0xae, 0xb0, 0x66, 0xa1, 0x8c, 0xea, 0xa1, 0x1c, 0x8b, 0x1a, 0x04,
            0x4f, 0x91, 0x32, 0x40, 0x8e, 0x87, 0x04, 0x8e, 0xb7, 0x41, 0xfe, 0x73, 0xed, 0x1b, 0xf6, 0x18,
        ],
        chunks: 856_040_456,
    },
];

/// The consensus-pinned PoM anchor for `model_id`, if it is a known PoM tier model.
pub fn pinned_pom_anchor(model_id: &[u8; 32]) -> Option<&'static PomAnchor> {
    POM_ANCHORS.iter().find(|a| &a.model_id == model_id)
}

// ── Legacy lineup (pre-OPoI-v2) ───────────────────────────────────────────────
// Served while `daa < OPOI_V2_ACTIVATION_DAA`. model_id values match the node's
// pre-v2 INFERENCE_REWARD_MINIMUMS table (8B/TinyLlama = CID-derived; 32B/70B =
// sha2-256(model.gguf) computed locally). Ported verbatim from the pre-rewrite
// registry so the transition is a true gate, not a re-derivation.

pub const TINYLLAMA: ModelSpec = ModelSpec {
    name: "tinyllama",
    // sha2-256(QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3)
    model_id: [
        0xe6, 0x4a, 0xf3, 0x68, 0xec, 0x93, 0x51, 0xa5,
        0xa4, 0xc0, 0xec, 0x7a, 0xe4, 0x7d, 0x42, 0xad,
        0xa7, 0xf6, 0xb3, 0xf1, 0xa6, 0xe6, 0x0f, 0xc7,
        0x3d, 0x0e, 0xb6, 0xca, 0x29, 0x53, 0x64, 0x5c,
    ],
    format: ModelFormat::Safetensors,
    tokenizer_cid: "QmSKrRu8HRt9v2dUeVdABKDkuREa5xFhPLZdevvvBfDYmp",
    config_cid: "QmbLTR3GLjBUKw8Lj14isiwG3XZJaL61ES852vkNqNPhyd",
    weight_cids: &["QmdqcmS8aMngiZWYYdeZEaW22N6XRTd9zK5ZCJG1MPmrQ3"],
    dir_name: "TinyLlama-1.1B",
    min_vram_mb: 0,
};

pub const DEEPSEEK_R1_8B: ModelSpec = ModelSpec {
    name: "deepseek-r1-8b",
    // sha2-256(QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR)
    model_id: [
        0x94, 0x29, 0x67, 0x33, 0x16, 0xbc, 0x40, 0xec,
        0x06, 0x67, 0x89, 0x45, 0x34, 0x57, 0x8b, 0x41,
        0x23, 0x6f, 0xc7, 0xee, 0xa4, 0xd9, 0x31, 0xf1,
        0x48, 0x9c, 0x34, 0xc5, 0x83, 0x7f, 0x42, 0xf4,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmXVdcr2FJuHtXcBbYbBuCMic2pJTkM1LJ6WpyfvhDytHg",
    config_cid: "",
    weight_cids: &["QmYK1faUGNMYZ2UKeSpUoUoFpRarZQEwfPCHbYNG2ib2mR"],
    dir_name: "DeepSeek-R1-8B",
    min_vram_mb: 5_500,
};

pub const DEEPSEEK_R1_32B: ModelSpec = ModelSpec {
    name: "deepseek-r1-32b",
    // sha2-256(model.gguf)
    model_id: [
        0xbe, 0xd9, 0xb0, 0xf5, 0x51, 0xf5, 0xb9, 0x5b,
        0xf9, 0xda, 0x58, 0x88, 0xa4, 0x8f, 0x0f, 0x87,
        0xc3, 0x7a, 0xd6, 0xb7, 0x25, 0x19, 0xc4, 0xcb,
        0xd7, 0x75, 0xf5, 0x4a, 0xc0, 0xb9, 0xfc, 0x62,
    ],
    format: ModelFormat::GgufQwen2,
    tokenizer_cid: "Qmf3uZwnuxZUhDbhup8Q51soVMRmNxohYctG9wZemNEPHm",
    config_cid: "",
    weight_cids: &["QmSrmkEoJUPf7r9t4o79F5APycnGrRu2icaU3KKPdFVUk7"],
    dir_name: "DeepSeek-R1-32B",
    min_vram_mb: 20_000,
};

pub const LLAMA_3_3_70B_OFFICIAL: ModelSpec = ModelSpec {
    name: "llama-3.3-70b-official",
    // sha2-256(model.gguf)
    model_id: [
        0xaa, 0xd2, 0xcf, 0x33, 0x48, 0xd8, 0xc7, 0xfd,
        0xbd, 0x2c, 0x0d, 0xd5, 0x8e, 0x0d, 0x99, 0x36,
        0x84, 0x50, 0xd4, 0x3c, 0x95, 0x84, 0xae, 0xf8,
        0x1a, 0x46, 0x7d, 0xd3, 0x47, 0x56, 0x13, 0x44,
    ],
    format: ModelFormat::Gguf,
    tokenizer_cid: "QmPd7WQvoQupfzpPVnVVc1Zra5SH4jKnGqNrdTHFtdQuvd",
    config_cid: "",
    weight_cids: &["QmbRQJFZ9NuZQW9uXezANTwunnwJCKybHiCFnVQ7D4SZKb"],
    // Distinct dir from the abliterated Llama-3.3-70B so both lineups coexist on disk.
    dir_name: "Llama-3.3-70B-official",
    min_vram_mb: 30_000,
};

/// OPoI v2 hardfork activation DAA score. MUST match the node's `opoi_v2_activation`.
/// Below this score the miner runs/announces the legacy lineup; at or above it, the
/// uncensored lineup. Mainnet: 37_780_000 (2026-06-26 18:00 UTC) — same H as the node's
/// MAINNET_PARAMS.opoi_v2_activation = new(37_780_000).
pub const OPOI_V2_ACTIVATION_DAA: u64 = 37_780_000;

/// H2 lineup-refresh hardfork activation DAA. MUST match the node's `very_light_activation`
/// (keryx-node MAINNET_PARAMS = new(38_951_445)). At this score the uncensored lineup changes:
///   - `--very-light` enters as PoM tier 0 (Qwen3-1.7B); before H2 it falls back to the light
///     tier (Gemma) so an early upgrader still mines a valid tier, and the existing tiers keep
///     their 4-tier indices.
///   - `--very-high` swaps Llama-3.3-70B Q4_K_M (48 GB-only) → Q2_K_L (fits a 32 GB 5090).
/// The node bundles a difficulty reset at this SAME DAA (the chain froze at pom_activation), so the
/// chain relaunches directly into the 5-tier scheme — the first re-mined blocks are H2 blocks.
/// Mainnet: 38_951_445, matching the node's `very_light_activation` in
/// `consensus/core/src/config/params.rs`. The network crossed this DAA ~2026-06-28, switching
/// the node to the 5-tier H2 `POM_TIERS_H2` table (Gemma moves from tier 0 to tier 1). A miner
/// still on `u64::MAX` here declares Gemma proofs as tier 0, which the node now verifies against
/// tier 0's (Qwen3-1.7B) root/chunks -> BadWeightPath on every share. See the CUDA fork's
/// identical fix (models.rs VERY_LIGHT_ACTIVATION_DAA).
/// (Named for very-light for history; it now gates the whole H2 refresh.)
pub const VERY_LIGHT_ACTIVATION_DAA: u64 = 38_951_445;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    VeryLight,
    Light,
    Default,
    High,
    VeryHigh,
}

/// Cumulative model set for a hardware tier within the lineup active at `daa`.
/// DAA-gated to mirror the node's `opoi_v2_activation`: one binary runs the legacy
/// lineup before H and the uncensored lineup at/after H (hot-swapped at the crossing),
/// so miners can upgrade before the hardfork without a flag-day restart.
pub fn specs_for(daa: u64, tier: Tier) -> &'static [&'static ModelSpec] {
    if daa >= OPOI_V2_ACTIVATION_DAA {
        // PoM era: one flag = one model. Each hardware tier mines AND serves exactly the
        // single model it proves possession of — the cumulative "serve everything below my
        // tier" behaviour is dropped, because a PoM GPU is bound to its tier (serving a
        // lower tier means unloading the mined model and pausing mining). Multi-tier
        // coverage is a network property (different miners per tier), not a per-GPU one.
        match tier {
            // Very-light only enters consensus at its own H2; before that it falls back to the
            // light tier (Gemma) so an early `--very-light` miner still mines a valid tier.
            Tier::VeryLight if daa >= VERY_LIGHT_ACTIVATION_DAA => &[&QWEN3_1_7B],
            Tier::VeryLight => &[&GEMMA_3_4B],
            Tier::Light => &[&GEMMA_3_4B],
            Tier::Default => &[&DOLPHIN_LLAMA3_8B],
            Tier::High => &[&QWEN3_32B],
            // At H2 the top tier swaps Q4_K_M (48 GB-only, effectively unserved) → Q2_K_L so it
            // fits a 32 GB 5090. Before H2 it stays the Q4 (live today).
            Tier::VeryHigh if daa >= VERY_LIGHT_ACTIVATION_DAA => &[&LLAMA_3_3_70B_Q2],
            Tier::VeryHigh => &[&LLAMA_3_3_70B],
        }
    } else {
        match tier {
            Tier::VeryLight | Tier::Light => &[&TINYLLAMA],
            Tier::Default => &[&TINYLLAMA, &DEEPSEEK_R1_8B],
            Tier::High => &[&TINYLLAMA, &DEEPSEEK_R1_8B, &DEEPSEEK_R1_32B],
            Tier::VeryHigh => &[&TINYLLAMA, &DEEPSEEK_R1_8B, &DEEPSEEK_R1_32B, &LLAMA_3_3_70B_OFFICIAL],
        }
    }
}

/// Both lineups combined — resolves a model name/id regardless of era.
pub const REGISTRY: &[&ModelSpec] = &[
    &GEMMA_3_4B,
    &DOLPHIN_LLAMA3_8B,
    &QWEN3_32B,
    &LLAMA_3_3_70B,
    &LLAMA_3_3_70B_Q2,
    &QWEN3_1_7B,
    &TINYLLAMA,
    &DEEPSEEK_R1_8B,
    &DEEPSEEK_R1_32B,
    &LLAMA_3_3_70B_OFFICIAL,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The H2 gate is consensus-critical: the tier index emitted in each PoM proof MUST match the
    // node's `pom_tiers(very_light_active)` per the block's own DAA. These lock the 4→5 tier reindex.
    const PRE_H2: u64 = VERY_LIGHT_ACTIVATION_DAA - 1;
    const AT_H2: u64 = VERY_LIGHT_ACTIVATION_DAA;

    #[test]
    fn pom_tier_index_pre_h2_is_4_tier() {
        assert_eq!(pom_tier_index(&GEMMA_3_4B.model_id, PRE_H2), Some(0));
        assert_eq!(pom_tier_index(&DOLPHIN_LLAMA3_8B.model_id, PRE_H2), Some(1));
        assert_eq!(pom_tier_index(&QWEN3_32B.model_id, PRE_H2), Some(2));
        assert_eq!(pom_tier_index(&LLAMA_3_3_70B.model_id, PRE_H2), Some(3));
        // The H2-only models are not in the pre-H2 tier set.
        assert_eq!(pom_tier_index(&QWEN3_1_7B.model_id, PRE_H2), None);
        assert_eq!(pom_tier_index(&LLAMA_3_3_70B_Q2.model_id, PRE_H2), None);
    }

    #[test]
    fn pom_tier_index_at_h2_is_5_tier() {
        assert_eq!(pom_tier_index(&QWEN3_1_7B.model_id, AT_H2), Some(0));
        assert_eq!(pom_tier_index(&GEMMA_3_4B.model_id, AT_H2), Some(1));
        assert_eq!(pom_tier_index(&DOLPHIN_LLAMA3_8B.model_id, AT_H2), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_32B.model_id, AT_H2), Some(3));
        assert_eq!(pom_tier_index(&LLAMA_3_3_70B_Q2.model_id, AT_H2), Some(4));
        // The Q4 70B is replaced by the Q2 at the top tier post-H2.
        assert_eq!(pom_tier_index(&LLAMA_3_3_70B.model_id, AT_H2), None);
    }

    #[test]
    fn is_pom_model_covers_both_eras() {
        for m in [&GEMMA_3_4B, &DOLPHIN_LLAMA3_8B, &QWEN3_32B, &LLAMA_3_3_70B, &QWEN3_1_7B, &LLAMA_3_3_70B_Q2] {
            assert!(is_pom_model(&m.model_id), "{} should be a PoM model", m.name);
        }
        assert!(!is_pom_model(&TINYLLAMA.model_id));
    }

    #[test]
    fn new_h2_anchors_resolve_with_expected_chunk_counts() {
        let q = pinned_pom_anchor(&QWEN3_1_7B.model_id).expect("Qwen3-1.7B anchor present");
        assert_eq!(q.chunks, 34_420_544);
        let l = pinned_pom_anchor(&LLAMA_3_3_70B_Q2.model_id).expect("Llama-70B-Q2 anchor present");
        assert_eq!(l.chunks, 856_040_456);
    }

    #[test]
    fn specs_for_swaps_at_h2_boundary() {
        // Very-light: Gemma fallback before its own H2, Qwen3-1.7B at/after.
        assert_eq!(specs_for(PRE_H2, Tier::VeryLight)[0].model_id, GEMMA_3_4B.model_id);
        assert_eq!(specs_for(AT_H2, Tier::VeryLight)[0].model_id, QWEN3_1_7B.model_id);
        // Very-high: Q4 before H2, Q2_K_L at/after.
        assert_eq!(specs_for(PRE_H2, Tier::VeryHigh)[0].model_id, LLAMA_3_3_70B.model_id);
        assert_eq!(specs_for(AT_H2, Tier::VeryHigh)[0].model_id, LLAMA_3_3_70B_Q2.model_id);
        // Light/default/high are unchanged across H2 (same model, only the tier INDEX shifts).
        assert_eq!(specs_for(PRE_H2, Tier::High)[0].model_id, specs_for(AT_H2, Tier::High)[0].model_id);
    }
}
