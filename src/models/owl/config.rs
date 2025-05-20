/// Model configuration for `OWLv2`
impl crate::Config {
    pub fn owlv2() -> Self {
        Self::default()
            .with_name("owlv2")
            // 1st & 3rd: text
            .with_model_ixx(0, 0, (1, 1, 1).into())
            .with_model_ixx(0, 1, 1.into())
            .with_model_ixx(2, 0, (1, 1, 1).into())
            .with_model_ixx(2, 1, 1.into())
            .with_model_max_length(16)
            // 2nd: image
            .with_model_ixx(1, 0, (1, 1, 1).into())
            .with_model_ixx(1, 1, 3.into())
            .with_model_ixx(1, 2, 960.into())
            .with_model_ixx(1, 3, 960.into())
            .with_image_mean(&[0.48145466, 0.4578275, 0.40821073])
            .with_image_std(&[0.26862954, 0.261_302_6, 0.275_777_1])
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_normalize(true)
            .with_class_confs(&[0.1])
            .with_model_num_dry_run(0)
            .with_tokenizer_file("owlv2/tokenizer.json")
    }

    pub fn owlv2_base() -> Self {
        Self::owlv2().with_model_file("base-patch16.onnx")
    }

    pub fn owlv2_base_ensemble() -> Self {
        Self::owlv2().with_model_file("base-patch16-ensemble.onnx")
    }

    pub fn owlv2_base_ft() -> Self {
        Self::owlv2().with_model_file("base-patch16-ft.onnx")
    }
}
