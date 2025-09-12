/// Model configuration for `DINOv3`
impl crate::Config {
    pub fn dinov3() -> Self {
        Self::dinov2().with_name("dinov3")
    }

    pub fn dinov3_vits16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vits16-pretrain-lvd1689m.onnx")
    }

    pub fn dinov3_vits16plus_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vits16plus-pretrain-lvd1689m.onnx")
    }

    pub fn dinov3_vitb16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vitb16-pretrain-lvd1689m.onnx")
    }

    pub fn dinov3_vitl16_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vitl16-pretrain-lvd1689m.onnx")
    }

    pub fn dinov3_vith16plus_lvd1689m() -> Self {
        Self::dinov3().with_model_file("vith16plus-pretrain-lvd1689m.onnx")
    }

    pub fn dinov3_vitl16_sat493m() -> Self {
        Self::dinov3().with_model_file("vitl16-pretrain-sat493m.onnx")
    }
}
