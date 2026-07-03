//! Phase-1 in-process OPoI inference — llama.cpp via FFI (`llama-cpp-2`, **Vulkan** backend).
//!
//! Drop-in replacement for the external `llama-server` child process (`llama_server.rs`):
//! same GGUF models, same ggml Vulkan backend, same greedy (temperature-0) decoding through
//! the model's own chat template — but linked into the miner, so there is no HTTP hop, no
//! child-process lifecycle, and (Phase 2) the PoM walk can eventually read inference's own
//! resident weight buffers instead of keeping a second VRAM copy.
//!
//! Compiled only with `--features inproc-llm`; `slm.rs` selects the engine at compile time.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use log::info;

/// Context window — matches the `-c 4096` the llama-server launch used.
const CTX_SIZE: u32 = 4096;

/// Process-wide ggml backend guard: `llama_backend_init` must run exactly once per process
/// (`LlamaBackend::init` errors on a second call). Never torn down — model switches drop the
/// [`LlamaEngine`] (freeing the model's VRAM) but keep the backend alive.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn backend() -> Result<&'static LlamaBackend> {
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _g = INIT.lock().unwrap_or_else(|p| p.into_inner());
    if BACKEND.get().is_none() {
        let b = LlamaBackend::init().map_err(|e| anyhow!("llama backend init failed: {e}"))?;
        let _ = BACKEND.set(b);
    }
    Ok(BACKEND.get().expect("backend just initialised"))
}

/// One resident model bound to the inference GPU, serving chat completions in-process.
/// The `slm.rs` engine seam (`launch` / `chat`) mirrors `LlamaServer` exactly.
pub struct LlamaEngine {
    model: LlamaModel,
    /// Serializes chats: each request runs a fresh short-lived context (its own KV cache),
    /// exactly like the stateless per-request usage of llama-server's chat endpoint.
    lock: Mutex<()>,
}

impl LlamaEngine {
    /// Load `gguf_path` fully onto the GPU (all layers — the in-process `-ngl 999`) and get
    /// ready to serve. Mirrors `LlamaServer::launch`, minus the child process and health poll.
    pub fn launch(gguf_path: &str) -> Result<Self> {
        // Multi-GPU rigs: pin ggml's Vulkan enumeration to the inference device BEFORE the
        // backend spins up, for the same reason llama_server.rs sets it on the child: left
        // alone, ggml layer-splits across every visible device, fighting the mining workers
        // on the other cards. Same loader, same order → the raw index maps 1:1. A user-set
        // value always wins.
        if std::env::var_os("GGML_VK_VISIBLE_DEVICES").is_none()
            && keryx_vulkan::enumerate_devices().len() > 1
        {
            let infer = keryx_vulkan::inference_device_index();
            info!("llm-engine: multi-GPU rig — pinning inference to Vulkan device {infer} (GGML_VK_VISIBLE_DEVICES)");
            std::env::set_var("GGML_VK_VISIBLE_DEVICES", infer.to_string());
        }

        let backend = backend()?;
        info!("llm-engine: loading {} (Vulkan, all layers on GPU, in-process)", gguf_path);
        // u32::MAX = offload every layer (the crate's "all" sentinel, like -ngl 999).
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(backend, Path::new(gguf_path), &params)
            .map_err(|e| anyhow!("llm-engine: model load failed: {e}"))?;
        info!("llm-engine: model resident, ready to serve");
        Ok(Self { model, lock: Mutex::new(()) })
    }

    /// Chat completion through the GGUF's own chat template, greedy decoding (temperature-0
    /// equivalent — keeps OPoI answers stable), capped at `max_tokens` generated tokens.
    pub fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let _serialize = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let backend = backend()?;

        // The model's baked-in chat template — the same one llama-server applies.
        let tmpl = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow!("llm-engine: model has no usable chat template: {e}"))?;
        let msgs = vec![
            LlamaChatMessage::new("system".to_string(), system.to_string())?,
            LlamaChatMessage::new("user".to_string(), user.to_string())?,
        ];
        // add_ass = true: end with the assistant header so generation starts the reply.
        let prompt = self.model.apply_chat_template(&tmpl, &msgs, true)?;

        // str_to_token parses special tokens (the template's control tokens) and AddBos
        // lets the tokenizer add BOS iff the model wants one — llama-server semantics.
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)?;
        if tokens.len() as u32 >= CTX_SIZE {
            return Err(anyhow!("llm-engine: prompt ({} tokens) exceeds the {} context", tokens.len(), CTX_SIZE));
        }

        // Fresh context per request: n_batch = CTX_SIZE so the whole prompt decodes in one
        // batch; KV is dropped with the context when this returns.
        let mut ctx = self.model.new_context(
            backend,
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(CTX_SIZE))
                .with_n_batch(CTX_SIZE),
        )?;

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        let last = tokens.len() as i32 - 1;
        for (i, tok) in (0_i32..).zip(tokens.iter().copied()) {
            batch.add(tok, i, &[0], i == last)?; // logits only for the last prompt token
        }
        ctx.decode(&mut batch)?;

        // Greedy decode until end-of-generation or the token budget. Output is accumulated
        // as BYTES: byte-level BPE tokens can split UTF-8 sequences mid-character, so
        // per-token string conversion would corrupt multi-byte output.
        let budget = max_tokens.min((CTX_SIZE as usize - tokens.len()).saturating_sub(1));
        let mut sampler = LlamaSampler::greedy();
        let mut out = Vec::<u8>::new();
        let mut n_cur = batch.n_tokens();
        for _ in 0..budget {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            // token_to_piece_bytes errors with the needed size when the hint is too small.
            let piece = match self.model.token_to_piece_bytes(token, 32, false, None) {
                Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(need)) => {
                    self.model.token_to_piece_bytes(token, (-need) as usize, false, None)
                }
                x => x,
            }?;
            out.extend_from_slice(&piece);
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end in-process inference on a real GGUF + GPU. Ignored by default (loads a
    /// multi-GB model); run with:
    ///   KERYX_TEST_GGUF=<path to model.gguf> cargo test --release --features inproc-llm -- --ignored inproc_chat
    #[test]
    #[ignore = "loads a multi-GB GGUF onto the GPU; set KERYX_TEST_GGUF to run"]
    fn inproc_chat_smoke() {
        let Ok(gguf) = std::env::var("KERYX_TEST_GGUF") else {
            eprintln!("SKIP: KERYX_TEST_GGUF not set");
            return;
        };
        let engine = LlamaEngine::launch(&gguf).expect("engine launch");
        let out = engine
            .chat("You are a terse assistant.", "Reply with the single word: pong", 16)
            .expect("chat");
        eprintln!("model replied: {out:?}");
        assert!(!out.trim().is_empty(), "empty completion");
        // Greedy decoding is deterministic: the same call must reproduce byte-identically.
        let again = engine
            .chat("You are a terse assistant.", "Reply with the single word: pong", 16)
            .expect("chat (repeat)");
        assert_eq!(out, again, "greedy decode not deterministic");
    }
}
