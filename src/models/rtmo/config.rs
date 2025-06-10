/// Model configuration for `RTMO`
impl crate::Config {
    pub fn rtmo() -> Self {
        Self::default()
            .with_name("rtmo")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_resize_filter("CatmullRom")
            .with_normalize(false)
            .with_nk(17)
            .with_class_confs(&[0.3])
            .with_keypoint_confs(&[0.5])
            .with_keypoint_names(&crate::NAMES_COCO_KEYPOINTS_17)
    }

    pub fn rtmo_t() -> Self {
        Self::rtmo()
            .with_model_ixx(0, 2, 416.into())
            .with_model_ixx(0, 3, 416.into())
            .with_model_file("t.onnx")
    }

    pub fn rtmo_s() -> Self {
        Self::rtmo().with_model_file("s.onnx")
    }

    pub fn rtmo_m() -> Self {
        Self::rtmo().with_model_file("m.onnx")
    }

    pub fn rtmo_l() -> Self {
        Self::rtmo().with_model_file("l.onnx")
    }
}
