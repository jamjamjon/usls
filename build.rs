use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["src/utils/onnx.proto3"], &["src"])?;

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");

    Ok(())
}
