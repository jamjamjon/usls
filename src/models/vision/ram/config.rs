///
/// > # RAM: Recognize Anything Model
/// >
/// > Open-world image recognition model capable of identifying thousands of object categories.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything)
/// > - **Paper RAM**: [Recognize Anything: A Strong Image Tagging Model](https://arxiv.org/abs/2306.03514)
/// > - **Paper RAM++**: [RAM++: Boosting Image Tagging with Pre-training and Data Augmentation](https://arxiv.org/abs/2310.15200)
/// >
/// > # Model Variants
/// >
/// > - **ram**: Base recognition model with 4585 categories
/// > - **ram-plus**: Enhanced model with improved performance
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Tagging**: Multi-label classification with 4585 categories
/// > - [X] **Bilingual Support**: Chinese and English category names
/// > - [X] **Open-world Recognition**: Broad object category coverage
/// >
/// Model configuration for `RAM`
///
impl crate::Config {
    /// Base recognition model with 4585 categories
    pub fn ram() -> Self {
        Self::default()
            .with_name("ram")
            .with_model_ixx(0, 0, (1, 1, 8))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 384)
            .with_model_ixx(0, 3, 384)
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_class_confs(&crate::CONFS_RAM_4585)
            .with_class_names(&crate::NAMES_RAM_ZH_4585)
            .with_class_names2(&crate::NAMES_RAM_EN_4585)
            .with_model_file("ram.onnx")
    }

    /// Enhanced model with improved performance
    pub fn ram_plus() -> Self {
        Self::ram().with_model_file("ram-plus.onnx")
    }
}
