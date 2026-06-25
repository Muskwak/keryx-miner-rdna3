//! Prebuilt llama.cpp **Vulkan** `llama-server` process manager + HTTP client.
//!
//! RDNA3 OPoI inference runs 100% on the GPU via Vulkan inside `llama-server` (all layers offloaded
//! with `-ngl 999`); the miner talks to it over localhost HTTP. No in-process LLM engine, no FFI,
//! no CUDA, no CPU inference. The server applies the GGUF's own chat template, so the miner only
//! supplies a system prompt + the user prompt and reads back the completion.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use log::info;

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8127;
const CTX_SIZE: u32 = 4096;
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Locate the `llama-server` binary: `$KERYX_LLAMA_SERVER`, then `<exe_dir>/llama/llama-server`,
/// then `<exe_dir>/llama-server`, then `PATH`.
fn locate_server() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    if let Ok(p) = std::env::var("KERYX_LLAMA_SERVER") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [dir.join("llama").join(exe_name), dir.join(exe_name)] {
                if cand.exists() {
                    return Some(cand);
                }
            }
        }
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe_name)).find(|p| p.exists())
}

/// True if a usable `llama-server` binary is present (used by the startup probe).
pub fn binary_available() -> bool {
    locate_server().is_some()
}

/// A running `llama-server` child process bound to one GGUF model, with localhost HTTP.
pub struct LlamaServer {
    child: Child,
    port: u16,
}

// Safe to share `&LlamaServer` across threads: `chat` is read-only (independent HTTP requests the
// server itself serialises) and never touches `child`; `child` is only mutated in `Drop`, which
// runs when the last `Arc` is released. Lets inference run without holding the engine lock.
unsafe impl Send for LlamaServer {}
unsafe impl Sync for LlamaServer {}

impl LlamaServer {
    /// Launch `llama-server` on `gguf_path` with full GPU (Vulkan) offload and wait until ready.
    pub fn launch(gguf_path: &str) -> Result<Self> {
        let bin = locate_server().ok_or_else(|| {
            anyhow!(
                "llama-server (Vulkan build) not found. Download the prebuilt llama.cpp Vulkan release \
                 (https://github.com/ggml-org/llama.cpp/releases — asset 'llama-*-bin-win-vulkan-x64.zip' \
                 on Windows) and either place llama-server[.exe] in '<miner_dir>/llama/' or point \
                 KERYX_LLAMA_SERVER at it."
            )
        })?;
        info!("llama-server: launching {} (Vulkan, all layers on GPU) for {}", bin.display(), gguf_path);
        let child = Command::new(&bin)
            .args([
                "-m",
                gguf_path,
                "-ngl",
                "999", // offload every layer to the GPU via Vulkan
                "--host",
                HOST,
                "--port",
                &PORT.to_string(),
                "-c",
                &CTX_SIZE.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("failed to spawn llama-server: {e}"))?;
        let mut server = Self { child, port: PORT };
        server.wait_until_ready()?;
        Ok(server)
    }

    fn base(&self) -> String {
        format!("http://{}:{}", HOST, self.port)
    }

    fn wait_until_ready(&mut self) -> Result<()> {
        let start = Instant::now();
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(anyhow!("llama-server exited during startup (status {status})"));
            }
            let healthy = ureq::get(&format!("{}/health", self.base()))
                .timeout(Duration::from_secs(2))
                .call()
                .map(|r| r.status() == 200)
                .unwrap_or(false);
            if healthy {
                info!("llama-server: ready on {}", self.base());
                return Ok(());
            }
            if start.elapsed() > LOAD_TIMEOUT {
                return Err(anyhow!("llama-server not ready after {:?}", LOAD_TIMEOUT));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Chat completion via the OpenAI-compatible endpoint. llama-server applies the model's chat
    /// template; greedy decoding (temperature 0) keeps OPoI answers stable.
    pub fn chat(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": false,
        });
        let resp = ureq::post(&format!("{}/v1/chat/completions", self.base()))
            .timeout(REQUEST_TIMEOUT)
            .send_json(body)
            .map_err(|e| anyhow!("chat request failed: {e}"))?;
        let v: serde_json::Value = resp.into_json().map_err(|e| anyhow!("chat response parse failed: {e}"))?;
        Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
