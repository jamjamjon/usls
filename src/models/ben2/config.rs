/// Model configuration for `BEN2`
impl crate::Config {
    pub fn ben2_base() -> Self {
        Self::rmbg().with_model_file("ben2-base.onnx")
    }
}
