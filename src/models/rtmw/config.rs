/// Model configuration for `RTMW`
impl crate::Config {
    pub fn rtmpose_133() -> Self {
        Self::rtmpose()
            .with_nk(133)
            .with_keypoint_names(&crate::NAMES_COCO_133)
    }

    pub fn rtmw_133_m() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-m.onnx")
    }

    pub fn rtmw_133_m_384() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-m-384.onnx")
    }

    pub fn rtmw_133_l() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-l.onnx")
    }

    pub fn rtmw_133_x() -> Self {
        Self::rtmpose_133().with_model_file("rtmw-133-x.onnx")
    }
}
