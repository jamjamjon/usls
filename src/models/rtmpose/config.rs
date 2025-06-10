/// Model configuration for `RTMPose`
impl crate::Config {
    pub fn rtmpose() -> Self {
        Self::default()
            .with_name("rtmpose")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 256.into())
            .with_model_ixx(0, 3, 192.into())
            .with_image_mean(&[123.675, 116.28, 103.53])
            .with_image_std(&[58.395, 57.12, 57.375])
            .with_normalize(false)
            .with_keypoint_confs(&[0.35])
    }

    pub fn rtmpose_17() -> Self {
        Self::rtmpose()
            .with_nk(17)
            .with_keypoint_names(&crate::NAMES_COCO_KEYPOINTS_17)
    }

    pub fn rtmpose_17_t() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-t.onnx")
    }

    pub fn rtmpose_17_s() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-s.onnx")
    }

    pub fn rtmpose_17_m() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-m.onnx")
    }

    pub fn rtmpose_17_l() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-l.onnx")
    }

    pub fn rtmpose_17_l_384() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-l-384.onnx")
    }

    pub fn rtmpose_17_x() -> Self {
        Self::rtmpose_17().with_model_file("rtmpose-17-x.onnx")
    }

    pub fn rtmpose_26() -> Self {
        Self::rtmpose()
            .with_nk(26)
            .with_keypoint_names(&crate::NAMES_HALPE_KEYPOINTS_26)
    }

    pub fn rtmpose_26_t() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-t.onnx")
    }

    pub fn rtmpose_26_s() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-s.onnx")
    }

    pub fn rtmpose_26_m() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-m.onnx")
    }

    pub fn rtmpose_26_m_384() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-m-384.onnx")
    }

    pub fn rtmpose_26_l() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-l.onnx")
    }

    pub fn rtmpose_26_l_384() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-l-384.onnx")
    }

    pub fn rtmpose_26_x() -> Self {
        Self::rtmpose_26().with_model_file("rtmpose-26-x.onnx")
    }
}
