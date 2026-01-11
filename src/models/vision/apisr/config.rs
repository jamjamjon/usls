///
/// > # APISR: Anime Production Inspired Real-World Anime Super-Resolution
/// >
/// > Super-resolution model for anime images inspired by anime production techniques.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [Kiteretsu77/APISR](https://github.com/Kiteretsu77/APISR)
/// >
/// > # Model Variants
/// >
/// > - **GRL-4x**: 4x super-resolution model
/// > - **RRDB-2x**: 2x super-resolution model
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Super-Resolution**: Anime image super-resolution
/// >
/// Model configuration for `APISR`
///
impl crate::Config {
    /// Base configuration for APISR models
    pub fn apisr() -> Self {
        Self::default()
            .with_name("apisr")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, (8, 8, 4096))
            .with_model_ixx(0, 3, (8, 8, 4096))
            .with_model_num_dry_run(0)
            .with_do_resize(false)
            .with_normalize(true)
            .with_pad_image(true)
            .with_pad_size(4)
    }

    /// 4x super-resolution model
    pub fn apisr_grl_4x() -> Self {
        Self::apisr()
            .with_sr_up_scale(4.)
            .with_model_file("GRL-4x.onnx")
    }

    /// 2x super-resolution model
    pub fn apisr_rrdb_2x() -> Self {
        Self::apisr()
            .with_sr_up_scale(2.)
            .with_model_file("RRDB-2x.onnx")
    }
}
