///
/// > # OWLv2: Scaling Open-Vocabulary Object Detection
/// >
/// > Open-vocabulary object detection model that can detect objects described by natural language without predefined classes.
/// >
/// > # Paper & Code
/// >
/// > - **Hugging Face**: [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble)
/// > - **Paper**: [OWLv2: Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
/// >
/// > # Model Variants
/// >
/// > - **owlv2-base**: Base OWLv2 model
/// > - **owlv2-base-ensemble**: Ensemble version for improved performance
/// > - **owlv2-base-ft**: Fine-tuned version
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Open-Vocabulary Detection**: Detect objects without predefined classes
/// > - [X] **Language-Guided Detection**: Use text descriptions to guide detection
/// > - [X] **High-Resolution Input**: 960x960 image processing
/// > - [X] **Ensemble Methods**: Multiple model ensembling for better accuracy
/// >
/// Model configuration for `OWLv2`
///
impl crate::Config {
    /// Base configuration for OWLv2 models
    pub fn owlv2() -> Self {
        Self::default()
            .with_name("owlv2")
            // 1st & 3rd: text
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 1)
            .with_model_ixx(2, 0, 1)
            .with_model_ixx(2, 1, 1)
            .with_model_max_length(16)
            // 2nd: image
            .with_model_ixx(1, 0, 1)
            .with_model_ixx(1, 1, 3)
            .with_model_ixx(1, 2, 960)
            .with_model_ixx(1, 3, 960)
            .with_image_mean([0.48145466, 0.4578275, 0.40821073])
            .with_image_std([0.26862954, 0.261_302_6, 0.275_777_1])
            .with_resize_mode_type(crate::ResizeModeType::FitAdaptive)
            .with_normalize(true)
            .with_class_confs(&[0.1])
            .with_model_num_dry_run(0)
            .with_tokenizer_file("owlv2/tokenizer.json")
    }

    /// Base OWLv2 model
    pub fn owlv2_base() -> Self {
        Self::owlv2().with_model_file("base-patch16.onnx")
    }

    /// Ensemble version for improved performance
    pub fn owlv2_base_ensemble() -> Self {
        Self::owlv2().with_model_file("base-patch16-ensemble.onnx")
    }

    /// Fine-tuned version
    pub fn owlv2_base_ft() -> Self {
        Self::owlv2().with_model_file("base-patch16-ft.onnx")
    }
}
