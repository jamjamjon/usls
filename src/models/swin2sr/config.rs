/// Model configuration for `Swin2SR`
impl crate::Config {
    pub fn swin2sr() -> Self {
        Self::default()
            .with_name("swin2sr")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, (8, 8, 4096).into())
            .with_model_ixx(0, 3, (8, 8, 4096).into())
            .with_model_num_dry_run(0)
            .with_do_resize(false)
            .with_normalize(true)
            .with_pad_image(true)
            .with_pad_size(8)
    }

    pub fn swin2sr_lightweight_x2_64() -> Self {
        Self::swin2sr()
            .with_up_scale(2.)
            .with_model_file("lightweight-x2-64.onnx")
    }

    pub fn swin2sr_classical_x2_64() -> Self {
        Self::swin2sr()
            .with_up_scale(2.)
            .with_model_file("classical-x2-64.onnx")
    }

    pub fn swin2sr_classical_x4_64() -> Self {
        Self::swin2sr()
            .with_up_scale(4.)
            .with_model_file("classical-x4-64.onnx")
    }

    pub fn swin2sr_realworld_x4_64_bsrgan_psnr() -> Self {
        Self::swin2sr()
            .with_up_scale(4.)
            .with_model_file("realworld-x4-64-bsrgan-psnr.onnx")
    }

    pub fn swin2sr_compressed_x4_48() -> Self {
        Self::swin2sr()
            .with_up_scale(4.)
            .with_model_file("compressed-x4-48.onnx")
    }
}
