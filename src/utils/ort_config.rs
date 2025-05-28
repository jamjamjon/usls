use aksr::Builder;
use anyhow::Result;

use crate::{try_fetch_file_stem, DType, Device, Hub, Iiix, MinOptMax};

#[derive(Builder, Debug, Clone)]
pub struct ORTConfig {
    pub file: String,
    pub device: Device,
    pub iiixs: Vec<Iiix>,
    pub num_dry_run: usize,
    pub trt_fp16: bool,
    pub graph_opt_level: Option<u8>,
    pub spec: String, // TODO: move out
    pub dtype: DType, // For dynamically loading the model
}

impl Default for ORTConfig {
    fn default() -> Self {
        Self {
            file: Default::default(),
            device: Default::default(),
            iiixs: Default::default(),
            graph_opt_level: Default::default(),
            spec: Default::default(),
            dtype: Default::default(),
            num_dry_run: 3,
            trt_fp16: true,
        }
    }
}

impl ORTConfig {
    pub fn try_commit(mut self, name: &str) -> Result<Self> {
        // Identify the local model or fetch the remote model
        if std::path::PathBuf::from(&self.file).exists() {
            // Local
            self.spec = format!("{}/{}", name, try_fetch_file_stem(&self.file)?);
        } else {
            if self.file.is_empty() && name.is_empty() {
                anyhow::bail!(
                    "Failed to commit model. Invalid model config: neither `name` nor `file` were specified. Failed to fetch model from Hub."
                )
            }

            // Remote
            match Hub::is_valid_github_release_url(&self.file) {
                Some((owner, repo, tag, _file_name)) => {
                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{}/{}-{}-{}-{}", name, owner, repo, tag, stem);
                    self.file = Hub::default().try_fetch(&self.file)?;
                }
                None => {
                    // append dtype to model file
                    match self.dtype {
                        d @ (DType::Auto | DType::Fp32) => {
                            if self.file.is_empty() {
                                self.file = format!("{}.onnx", d);
                            }
                        }
                        dtype => {
                            if self.file.is_empty() {
                                self.file = format!("{}.onnx", dtype);
                            } else {
                                let pos = self.file.len() - 5; // .onnx
                                let suffix = self.file.split_off(pos);
                                self.file = format!("{}-{}{}", self.file, dtype, suffix);
                            }
                        }
                    }

                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{}/{}", name, stem);
                    self.file = Hub::default().try_fetch(&format!("{}/{}", name, self.file))?;

                    // try fetch external data file if it exists
                    match Hub::default().try_fetch(&format!("{}_data", self.file)) {
                        Ok(external_data_file) => {
                            log::debug!(
                                "Successfully fetched external data file: {}",
                                external_data_file
                            );
                        }
                        Err(_) => {
                            log::debug!("No external data file found for model {}", self.file);
                        }
                    }
                }
            }
        }

        Ok(self)
    }
}

impl ORTConfig {
    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }

    pub fn with_batch_size(mut self, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((0, 0, x)));
        self
    }
}

#[macro_export]
macro_rules! impl_ort_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            paste::paste! {
                pub fn [<with_ $field _file>](mut self, file: &str) -> Self {
                    self.$field = self.$field.with_file(file);
                    self
                }
                pub fn [<with_ $field _dtype>](mut self, dtype: $crate::DType) -> Self {
                    self.$field = self.$field.with_dtype(dtype);
                    self
                }
                pub fn [<with_ $field _device>](mut self, device: $crate::Device) -> Self {
                    self.$field = self.$field.with_device(device);
                    self
                }
                pub fn [<with_ $field _trt_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_trt_fp16(x);
                    self
                }
                pub fn [<with_ $field _num_dry_run>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_dry_run(x);
                    self
                }
                pub fn [<with_ $field _ixx>](mut self, i: usize, ii: usize, x: $crate::MinOptMax) -> Self {
                    self.$field = self.$field.with_ixx(i, ii, x);
                    self
                }
            }
        }
    };
}
