/// Model configuration for `YOLOP`
impl crate::Config {
    pub fn yolop() -> Self {
        Self::default()
            .with_name("yolop")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 2, 640.into())
            .with_model_ixx(0, 3, 640.into())
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_class_confs(&[0.3])
    }

    pub fn yolop_v2_480x800() -> Self {
        Self::yolop().with_model_file("v2-480x800.onnx")
    }

    pub fn yolop_v2_736x1280() -> Self {
        Self::yolop().with_model_file("v2-736x1280.onnx")
    }
}
