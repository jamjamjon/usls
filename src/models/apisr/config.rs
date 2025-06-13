/// Model configuration for `APISR`
impl crate::Config {
    pub fn apisr() -> Self {
        Self::default()
            .with_name("apisr")
            .with_model_ixx(0, 0, 1.into())
            .with_model_ixx(0, 1, 3.into())
            .with_model_ixx(0, 2, (8, 8, 4096).into())
            .with_model_ixx(0, 3, (8, 8, 4096).into())
            .with_model_num_dry_run(0)
            .with_do_resize(false)
            .with_normalize(true)
            .with_pad_image(true)
            .with_pad_size(4)
    }

    pub fn apisr_grl_4x() -> Self {
        Self::apisr()
            .with_up_scale(4.)
            .with_model_file("GRL-4x.onnx")
    }

    pub fn apisr_rrdb_2x() -> Self {
        Self::apisr()
            .with_up_scale(2.)
            .with_model_file("RRDB-2x.onnx")
    }
}
