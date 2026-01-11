///
/// > # SLANet: Scene Layout Analysis Network
/// >
/// > Table layout analysis network for document structure recognition and text detection.
/// >
/// > # Paper & Code
/// >
/// > - **Official**: [PaddleOCR SLANet](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html)
/// >
/// > # Model Variants
/// >
/// > - **slanet-lcnet-v2-mobile-ch**: Mobile-optimized model for Chinese text analysis
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Table Layout Analysis**: Document structure recognition
/// > - [X] **Scene Text Detection**: Text detection in complex scenes
/// > - [X] **Mobile Optimization**: Efficient inference on mobile devices
/// > - [X] **Chinese Text Support**: Optimized for Chinese character recognition
/// >
/// Model configuration for `SLANet`
///
impl crate::Config {
    /// Base configuration for SLANet models
    pub fn slanet() -> Self {
        Self::default()
            .with_name("slanet")
            .with_model_ixx(0, 0, (1, 1, 8))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, (320, 488, 488))
            .with_model_ixx(0, 3, (320, 488, 488))
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_padding_value(0)
            .with_unsigned(true)
    }

    /// Mobile-optimized model for Chinese text analysis
    pub fn slanet_lcnet_v2_mobile_ch() -> Self {
        Self::slanet()
            .with_model_file("v2-mobile-ch.onnx")
            .with_class_names_owned(Self::load_txt_into_vec("slanet/vocab-sla-v2.txt"))
    }
}
