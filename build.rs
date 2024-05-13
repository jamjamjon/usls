fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Need this for CoreML. See: https://ort.pyke.io/perf/execution-providers#coreml
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");

    // Build onnx
    prost_build::compile_protos(&["src/core/onnx.proto3"], &["src/core"])?;
    Ok(())
}
