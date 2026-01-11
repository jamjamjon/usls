use crate::{Config, DType, Device, Iiix};

impl Config {
    /// Apply num_dry_run to all modules.
    pub fn with_num_dry_run_all(mut self, x: usize) -> Self {
        for config in self.modules.values_mut() {
            config.num_dry_run = x;
        }
        self
    }

    /// Apply batch size (min, opt, max) to all modules.
    pub fn with_batch_size_all_min_opt_max(mut self, min: usize, opt: usize, max: usize) -> Self {
        for config in self.modules.values_mut() {
            config
                .iiixs
                .push(Iiix::from((0, 0, (min, opt, max).into())));
        }
        self
    }

    /// Apply batch size to all modules.
    pub fn with_batch_size_all(mut self, batch_size: usize) -> Self {
        for config in self.modules.values_mut() {
            config.iiixs.push(Iiix::from((0, 0, batch_size.into())));
        }
        self
    }

    /// Apply device to all modules.
    pub fn with_device_all(mut self, device: Device) -> Self {
        for config in self.modules.values_mut() {
            config.device = device;
        }
        self
    }

    /// Apply dtype to all modules.
    pub fn with_dtype_all(mut self, dtype: DType) -> Self {
        for config in self.modules.values_mut() {
            config.dtype = dtype;
        }
        self
    }

    /// Apply graph optimization level to all modules.
    pub fn with_graph_opt_level_all(mut self, level: u8) -> Self {
        for config in self.modules.values_mut() {
            config.graph_opt_level = Some(level);
        }
        self
    }

    /// Apply intra threads to all modules.
    pub fn with_num_intra_threads_all(mut self, num_threads: usize) -> Self {
        for config in self.modules.values_mut() {
            config.num_intra_threads = Some(num_threads);
        }
        self
    }

    /// Apply inter threads to all modules.
    pub fn with_num_inter_threads_all(mut self, num_threads: usize) -> Self {
        for config in self.modules.values_mut() {
            config.num_inter_threads = Some(num_threads);
        }
        self
    }
}
