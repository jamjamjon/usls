use aksr::Builder;
use anyhow::Result;

use crate::{try_fetch_file_stem, DType, Device, Hub, Iiix, Kind, MinOptMax, Scale, Task, Version};

#[derive(Builder, Debug, Clone, Default)]
pub struct ModelConfig {
    // engine needs this
    pub file: String,
    pub device: Device, // optional
    pub iiixs: Vec<Iiix>,
    pub spec: String,
    pub num_dry_run: usize,
    pub trt_fp16: bool,
    pub ort_graph_opt_level: Option<u8>,

    // ---
    pub name: &'static str,
    pub dtype: DType,
    pub version: Option<Version>, // optional
    pub task: Option<Task>,
    pub scale: Option<Scale>,
    pub kind: Option<Kind>, // TODO: remove  // optional
}

impl ModelConfig {
    pub fn commit(mut self) -> Result<Self> {
        // Identify the local model or fetch the remote model
        if std::path::PathBuf::from(&self.file).exists() {
            // Local
            self.spec = format!("{}/{}", self.name, try_fetch_file_stem(&self.file)?);
        } else {
            // Remote
            if self.file.is_empty() && self.name.is_empty() {
                anyhow::bail!(
                    "Neither `name` nor `file` were specified. Faild to fetch model from remote."
                )
            }

            // Load
            match Hub::is_valid_github_release_url(&self.file) {
                Some((owner, repo, tag, _file_name)) => {
                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{}/{}-{}-{}-{}", self.name, owner, repo, tag, stem);
                    self.file = Hub::default().try_fetch(&self.file)?;
                }
                None => {
                    // special yolo case
                    if self.file.is_empty() && self.name == "yolo" {
                        // [version]-[scale]-[task]
                        let mut y = String::new();
                        if let Some(x) = self.version() {
                            y.push_str(&x.to_string());
                        }
                        if let Some(x) = self.scale() {
                            y.push_str(&format!("-{}", x));
                        }
                        if let Some(x) = self.task() {
                            y.push_str(&format!("-{}", x.yolo_str()));
                        }
                        y.push_str(".onnx");
                        self.file = y;
                    }

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
                    self.spec = format!("{}/{}", self.name, stem);
                    self.file =
                        Hub::default().try_fetch(&format!("{}/{}", self.name, self.file))?;
                }
            }
        }

        Ok(self)
    }
}

impl ModelConfig {
    pub fn with_ixx(mut self, i: usize, ii: usize, x: MinOptMax) -> Self {
        self.iiixs.push(Iiix::from((i, ii, x)));
        self
    }
}

#[macro_export]
macro_rules! impl_model_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            paste::paste! {
                pub fn [<with_ $field _name>](mut self, name: &'static str) -> Self {
                    self.$field = self.$field.with_name(name);
                    self
                }
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
                pub fn [<with_ $field _version>](mut self, version: $crate::Version) -> Self {
                    self.$field = self.$field.with_version(version);
                    self
                }
                pub fn [<with_ $field _task>](mut self, task: $crate::Task) -> Self {
                    self.$field = self.$field.with_task(task);
                    self
                }
                pub fn [<with_ $field _scale>](mut self, scale: $crate::Scale) -> Self {
                    self.$field = self.$field.with_scale(scale);
                    self
                }
                pub fn [<with_ $field _kind>](mut self, kind: $crate::Kind) -> Self {
                    self.$field = self.$field.with_kind(kind);
                    self
                }
                pub fn [<with_ $field _ixx>](mut self, i: usize, ii: usize, x: $crate::MinOptMax) -> Self {
                    self.$field = self.$field.with_ixx(i, ii, x);
                    self
                }
                // pub fn [<with_ $field _xx>](mut self) -> anyhow::Result<Self> {
                //     self.$field = self.$field.commit()?;
                //     Ok(self)
                // }
            }


        }
    };
}
