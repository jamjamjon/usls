/// Model configuration for `DWPose`
impl crate::Config {
    pub fn dwpose_133_t() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-t.onnx")
    }

    pub fn dwpose_133_s() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-s.onnx")
    }

    pub fn dwpose_133_m() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-m.onnx")
    }

    pub fn dwpose_133_l() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-l.onnx")
    }

    pub fn dwpose_133_l_384() -> Self {
        Self::rtmpose_133().with_model_file("dwpose-133-l-384.onnx")
    }
}
