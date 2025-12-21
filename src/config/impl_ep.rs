impl crate::Config {
    /// Apply CPU arena allocator to all modules.
    pub fn with_cpu_arena_allocator_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.cpu.arena_allocator = x;
        }
        self
    }

    /// Apply OpenVINO dynamic shapes to all modules.
    pub fn with_openvino_dynamic_shapes_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.openvino.dynamic_shapes = x;
        }
        self
    }

    /// Apply OpenVINO OpenCL throttling to all modules.
    pub fn with_openvino_opencl_throttling_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.openvino.opencl_throttling = x;
        }
        self
    }

    /// Apply OpenVINO QDQ optimizer to all modules.
    pub fn with_openvino_qdq_optimizer_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.openvino.qdq_optimizer = x;
        }
        self
    }

    /// Apply OpenVINO num threads to all modules.
    pub fn with_openvino_num_threads_all(mut self, num_threads: usize) -> Self {
        for config in self.modules.values_mut() {
            config.ep.openvino.num_threads = Some(num_threads);
        }
        self
    }

    /// Apply OneDNN arena allocator to all modules.
    pub fn with_onednn_arena_allocator_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.onednn.arena_allocator = x;
        }
        self
    }

    /// Apply TensorRT FP16 to all modules.
    pub fn with_tensorrt_fp16_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.tensorrt.fp16 = x;
        }
        self
    }

    /// Apply TensorRT engine cache to all modules.
    pub fn with_tensorrt_engine_cache_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.tensorrt.engine_cache = x;
        }
        self
    }

    /// Apply TensorRT timing cache to all modules.
    pub fn with_tensorrt_timing_cache_all(mut self, x: bool) -> Self {
        for config in self.modules.values_mut() {
            config.ep.tensorrt.timing_cache = x;
        }
        self
    }

    /// Set TensorRT FP16 for the Model engine.
    pub fn with_model_tensorrt_fp16(mut self, x: bool) -> Self {
        if let Some(engine) = self.modules.get_mut(&crate::Module::Model) {
            engine.ep.tensorrt.fp16 = x;
        }
        self
    }
}
