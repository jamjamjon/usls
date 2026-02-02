///
/// > # PP-DocLayout: Document Layout Analysis
/// >
/// > PaddlePaddle's document layout analysis model for detecting various document elements
/// > such as text, tables, figures, headers, footers, etc. in document images.
/// >
/// > # Paper & Code
/// >
/// > - **v3 Model**: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3
/// > - **v2 Model**: https://huggingface.co/PaddlePaddle/PP-DocLayoutV2
/// > - **v1 Model**: https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L
/// > - **GitHub**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
/// >
/// > # Model Variants
/// >
/// > - **PP-DocLayout-Plus-L**: 20 classes for basic document layout analysis
/// > - **PP-DocLayoutV2**: 25 classes with enhanced detection and reading order support
/// >
/// Model configuration for `PPDocLayout`
///
impl crate::Config {
    /// Base configuration for PP-DocLayout
    pub fn pp_doclayout() -> Self {
        Self::default()
            .with_name("pp-doclayout")
            .with_batch_size_min_opt_max_all(1, 1, 8)
            .with_model_ixx(0, 1, 2) // img shapes
            .with_model_ixx(1, 1, 3)
            .with_model_ixx(1, 2, 800)
            .with_model_ixx(1, 3, 800)
            .with_model_ixx(2, 1, 2) // scale factors
            .with_class_confs(&[0.35])
            .with_resize_alg(crate::ResizeAlg::Interpolation(
                crate::ResizeFilter::Bilinear,
            ))
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
    }

    /// PP-DocLayout-Plus-L configuration (20 classes)
    pub fn pp_doclayout_v1_plus_l() -> Self {
        Self::pp_doclayout()
            .with_version(1.into())
            .with_class_names(&crate::NAMES_PP_DOC_LAYOUT_V1_20)
            .with_model_file("v1-plus-l.onnx")
    }

    /// PP-DocLayoutV2 configuration (25 classes with reading order)
    pub fn pp_doclayout_v2() -> Self {
        Self::pp_doclayout()
            .with_version(2.into())
            .with_class_names(&crate::NAMES_PP_DOC_LAYOUT_V2_25)
            .with_model_file("v2.onnx")
    }

    /// PP-DocLayoutV3 configuration (25 classes with reading order)
    /// PP-DocLayoutV3 is specifically engineered to handle non-planar document images.
    /// It can directly predict multi-point bounding boxes for layout elements—as opposed to standard two-point boxes—and determine logical reading orders for skewed and curved surfaces within a single forward pass, significantly reducing cascading errors.
    pub fn pp_doclayout_v3() -> Self {
        Self::pp_doclayout()
            .with_version(3.into())
            .with_class_names(&crate::NAMES_PP_DOC_LAYOUT_V2_25)
            .with_model_file("v3.onnx")
    }
}
