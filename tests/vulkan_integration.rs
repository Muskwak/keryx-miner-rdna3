//! Smoke test: the integrated miner crate detects the RDNA3 Vulkan device through the same probe
//! the startup path uses. Confirms keryx-vulkan is wired into keryx-miner and runs on this GPU.

#[test]
fn vulkan_device_detected_through_integration() {
    match keryx_miner::slm::probe_gpu_inference() {
        // Ok (llama-server also present) or NoServer (binary absent) both mean the Vulkan device
        // was found — which is what this test asserts. Only NoDevice is a failure.
        keryx_miner::slm::GpuProbe::Ok | keryx_miner::slm::GpuProbe::NoServer => {}
        keryx_miner::slm::GpuProbe::NoDevice => {
            panic!("no Vulkan device detected via the integrated probe");
        }
    }
}
