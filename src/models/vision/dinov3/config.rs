///
/// > # DINOv3: Advanced Vision Transformer Features
/// >
/// > Next-generation self-supervised vision transformer with enhanced performance.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
/// > - **Paper**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07793)
///
/// > # Model Variants
/// >
/// > - **dinov3-vits16-lvd1689m**: Small model with 1.689B parameters
/// > - **dinov3-vits16plus-lvd1689m**: Small+ model with 1.689B parameters
/// > - **dinov3-vitb16-lvd1689m**: Base model with 1.689B parameters
/// > - **dinov3-vitl16-lvd1689m**: Large model with 1.689B parameters
/// > - **dinov3-vith16plus-lvd1689m**: Large+ model with 1.689B parameters
/// > - **dinov3-vitl16-sat493m**: Large model with 493M parameters
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Advanced Feature Extraction**: Enhanced self-supervised learning
/// > - [X] **Multiple Scales**: Various model sizes for different use cases
/// >
/// Model configuration for `DINOv3`
///
impl crate::Config {
    /// Base configuration for DINOv3 models
    pub fn dinov3() -> Self {
        Self::dinov2().with_name("dinov3")
    }

    /// Small model with 1.689B parameters
    pub fn dinov3_vits16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vits16-pretrain-lvd1689m.onnx")
    }

    /// Small+ model with 1.689B parameters
    pub fn dinov3_vits16plus_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vits16plus-pretrain-lvd1689m.onnx")
    }

    /// Base model with 1.689B parameters
    pub fn dinov3_vitb16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vitb16-pretrain-lvd1689m.onnx")
    }

    /// Large model with 1.689B parameters
    pub fn dinov3_vitl16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vitl16-pretrain-lvd1689m.onnx")
    }

    /// Large+ model with 1.689B parameters
    pub fn dinov3_vith16plus_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vith16plus-pretrain-lvd1689m.onnx")
    }

    /// Large model with 493M parameters
    pub fn dinov3_vitl16_sat493m() -> Self {
        Self::dinov3().with_model_file("vitl16-pretrain-sat493m.onnx")
    }
}
