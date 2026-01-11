///
/// > # Swin2SR: SwinV2 Transformer for Super-Resolution
/// >
/// > SwinV2 transformer for compressed image super-resolution and restoration tasks.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr)
/// > - **Paper**: [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345)
/// >
/// > # Model Variants
/// >
/// > - **swin2sr-lightweight-x2-64**: Lightweight 2x super-resolution model
/// > - **swin2sr-classical-x2-64**: Classical 2x super-resolution model
/// > - **swin2sr-classical-x4-64**: Classical 4x super-resolution model
/// > - **swin2sr-realworld-x4-64**: Real-world 4x super-resolution model
/// > - **swin2sr-compressed-x4-48**: Compressed image 4x super-resolution model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Image Super-Resolution**: 2x and 4x image upscaling
/// > - [X] **Compressed Image Restoration**: Restore compressed images
/// > - [X] **Real-world Enhancement**: Real-world image super-resolution
/// > - [X] **Transformer Architecture**: SwinV2 backbone for performance
/// >
/// Model configuration for `Swin2SR`
///
impl crate::Config {
    /// Base configuration for Swin2SR models
    pub fn swin2sr() -> Self {
        Self::default()
            .with_name("swin2sr")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, (8, 8, 4096))
            .with_model_ixx(0, 3, (8, 8, 4096))
            .with_model_num_dry_run(0)
            .with_do_resize(false)
            .with_normalize(true)
            .with_pad_image(true)
            .with_pad_size(8)
    }

    /// Lightweight 2x super-resolution model
    pub fn swin2sr_lightweight_x2_64() -> Self {
        Self::swin2sr()
            .with_sr_up_scale(2.)
            .with_model_file("lightweight-x2-64.onnx")
    }

    /// Classical 2x super-resolution model
    pub fn swin2sr_classical_x2_64() -> Self {
        Self::swin2sr()
            .with_sr_up_scale(2.)
            .with_model_file("classical-x2-64.onnx")
    }

    /// Classical 4x super-resolution model
    pub fn swin2sr_classical_x4_64() -> Self {
        Self::swin2sr()
            .with_sr_up_scale(4.)
            .with_model_file("classical-x4-64.onnx")
    }

    /// Real-world 4x super-resolution model
    pub fn swin2sr_realworld_x4_64_bsrgan_psnr() -> Self {
        Self::swin2sr()
            .with_sr_up_scale(4.)
            .with_model_file("realworld-x4-64-bsrgan-psnr.onnx")
    }

    /// Compressed image 4x super-resolution model
    pub fn swin2sr_compressed_x4_48() -> Self {
        Self::swin2sr()
            .with_sr_up_scale(4.)
            .with_model_file("compressed-x4-48.onnx")
    }
}
