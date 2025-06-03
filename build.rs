use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["src/utils/onnx.proto3"], &["src"])?;

    Ok(())
}
