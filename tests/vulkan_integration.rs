//! Smoke test: the integrated miner crate detects the RDNA3 Vulkan device through the same probe
//! the startup path uses. Confirms keryx-vulkan is wired into keryx-miner and runs on this GPU.

#[test]
fn vulkan_device_detected_through_integration() {
    match keryx_miner::slm::probe_gpu_inference() {
        // The engine is linked in-process, so Ok simply means the Vulkan device was found —
        // which is what this test asserts. Only NoDevice is a failure.
        keryx_miner::slm::GpuProbe::Ok => {}
        keryx_miner::slm::GpuProbe::NoDevice => {
            // CI runners (GitHub Actions sets CI=true) have no GPU — skip rather than fail there,
            // matching the keryx-vulkan tests. On real hardware CI is unset, so a missing device
            // is still a hard failure and this stays a meaningful smoke test.
            if std::env::var_os("CI").is_some() {
                eprintln!("SKIP: no Vulkan device in CI container (GitHub runners have no GPU)");
                return;
            }
            panic!("no Vulkan device detected via the integrated probe");
        }
    }
}
