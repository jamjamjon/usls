/// Model configuration for `SLANet`
impl crate::Config {
    pub fn slanet() -> Self {
        Self::default()
            .with_name("slanet")
            .with_model_ixx(0, 0, (1, 1, 8).into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, (320, 488, 488).into())
            .with_model_ixx(0, 3, (320, 488, 488).into())
            .with_image_mean(&[0.485, 0.456, 0.406])
            .with_image_std(&[0.229, 0.224, 0.225])
            .with_resize_mode(crate::ResizeMode::FitAdaptive)
            .with_padding_value(0)
            .with_unsigned(true)
    }

    pub fn slanet_lcnet_v2_mobile_ch() -> Self {
        Self::slanet()
            .with_model_file("v2-mobile-ch.onnx")
            .with_vocab_txt("slanet/vocab-sla-v2.txt")
    }
}
