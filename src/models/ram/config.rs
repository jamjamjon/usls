/// Model configuration for [RAM](https://arxiv.org/abs/2306.03514) and [RAM++](https://arxiv.org/abs/2310.15200)
impl crate::Config {
    pub fn ram() -> Self {
        Self::default()
            .with_name("ram")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 384.into())
            .with_model_ixx(0, 3, 384.into())
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_class_confs(&crate::CONFS_RAM_4585)
            .with_class_names(&crate::NAMES_RAM_ZH_4585)
            .with_class_names2(&crate::NAMES_RAM_EN_4585)
            .with_model_file("ram.onnx")
    }

    pub fn ram_plus() -> Self {
        Self::ram().with_model_file("ram-plus.onnx")
    }
}
