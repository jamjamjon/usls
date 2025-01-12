/// Model configuration for `DepthPro`
impl crate::Options {
    pub fn depth_pro() -> Self {
        Self::default()
            .with_model_name("depth-pro")
            .with_model_ixx(0, 0, 1.into()) // batch. Note: now only support batch_size = 1
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, 1536.into())
            .with_model_ixx(0, 3, 1536.into())
            .with_image_mean(&[0.5, 0.5, 0.5])
            .with_image_std(&[0.5, 0.5, 0.5])
            .with_resize_mode(crate::ResizeMode::FitExact)
            .with_normalize(true)
    }

    // pub fn depth_pro_q4f16() -> Self {
    //     Self::depth_pro().with_model_file("q4f16.onnx")
    // }

    // pub fn depth_pro_fp16() -> Self {
    //     Self::depth_pro().with_model_file("fp16.onnx")
    // }

    // pub fn depth_pro_bnb4() -> Self {
    //     Self::depth_pro().with_model_file("bnb4.onnx")
    // }
}
