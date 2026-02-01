use crate::Config;

// ===== Macros for generating EP configuration functions =====

/// Generate functions for specific modules (Model, Visual, Textual, Encoder).
///
/// Pattern: `with_<module>_<ep>_<field>()`
///
/// # Parameters
/// - `$ep`: Execution provider name
/// - `$field`: Configuration field name
/// - `$ty`: Field type
/// - `$module`: Module names (variadic)
macro_rules! impl_ep_for_modules {
    ($ep:ident, $field:ident, $ty:ty, $($module:ident),+) => {
        $(
            paste::paste! {
                #[doc = "Set `" $ep "." $field "` for the `" $module "` module."]
                pub fn [<with_ $module:snake _ $ep _ $field>](mut self, x: $ty) -> Self {
                    if let Some(config) = self.modules.get_mut(&crate::Module::$module) {
                        config.ep.$ep.$field = x;
                    }
                    self
                }
            }
        )+
    };
}

/// Generate function for a specific module parameter.
///
/// Pattern: `with_<ep>_<field>_module()`
///
/// # Parameters
/// - `$ep`: Execution provider name
/// - `$field`: Configuration field name
/// - `$ty`: Field type
macro_rules! impl_ep_for_module {
    ($ep:ident, $field:ident, $ty:ty) => {
        paste::paste! {
            #[doc = "Set `" $ep "." $field "` for a specific module."]
            pub fn [<with_ $ep _ $field _ module>](mut self, module: crate::Module, x: $ty) -> Self {
                if let Some(config) = self.modules.get_mut(&module) {
                    config.ep.$ep.$field = x;
                }
                self
            }
        }
    };
}

/// Generate function for all modules.
///
/// Pattern: `with_<ep>_<field>_all()`
///
/// # Parameters
/// - `$ep`: Execution provider name
/// - `$field`: Configuration field name
/// - `$ty`: Field type
macro_rules! impl_ep_for_all {
    ($ep:ident, $field:ident, $ty:ty) => {
        paste::paste! {
            #[doc = "Apply `" $ep "." $field "` to all modules."]
            pub fn [<with_ $ep _ $field _ all>](mut self, x: $ty) -> Self {
                for config in self.modules.values_mut() {
                    config.ep.$ep.$field = x;
                }
                self
            }
        }
    };
}

/// Generate all three patterns for an EP field.
///
/// This macro combines [`impl_ep_for_modules!`], [`impl_ep_for_module!`], and [`impl_ep_for_all!`]
/// to generate a complete set of configuration functions for a given execution provider field.
///
/// # Parameters
/// - `$ep`: Execution provider name
/// - `$field`: Configuration field name
/// - `$ty`: Field type
macro_rules! impl_ep_field {
    ($ep:ident, $field:ident, $ty:ty) => {
        impl_ep_for_modules!(
            $ep,
            $field,
            $ty,
            Model,
            Visual,
            Textual,
            Encoder,
            Decoder,
            VisualEncoder,
            TextualEncoder,
            VisualDecoder,
            TextualDecoder,
            TextualDecoderMerged,
            VisualProjection,
            TextualProjection
        );
        impl_ep_for_module!($ep, $field, $ty);
        impl_ep_for_all!($ep, $field, $ty);
    };
}

impl Config {
    // ---------------- CPU -----------------------
    impl_ep_field!(cpu, arena_allocator, bool);

    // ---------------- CUDA -----------------------
    impl_ep_field!(cuda, cuda_graph, bool);
    impl_ep_field!(cuda, fuse_conv_bias, bool);
    impl_ep_field!(cuda, conv_max_workspace, bool);
    impl_ep_field!(cuda, tf32, bool);
    impl_ep_field!(cuda, prefer_nhwc, bool);

    // ---------------- TensorRT -----------------------
    impl_ep_field!(tensorrt, dump_ep_context_model, bool);
    impl_ep_field!(tensorrt, dump_subgraphs, bool);
    impl_ep_field!(tensorrt, builder_optimization_level, u8);
    impl_ep_field!(tensorrt, max_workspace_size, usize);
    impl_ep_field!(tensorrt, min_subgraph_size, usize);
    impl_ep_field!(tensorrt, fp16, bool);
    impl_ep_field!(tensorrt, int8, bool);
    impl_ep_field!(tensorrt, engine_cache, bool);
    impl_ep_field!(tensorrt, timing_cache, bool);
    impl_ep_field!(tensorrt, dla, bool);
    impl_ep_field!(tensorrt, dla_core, u32);
    impl_ep_field!(tensorrt, int8_use_native_calibration_table, bool);
    impl_ep_field!(tensorrt, detailed_build_log, bool);

    // ---------------- coreml -----------------------
    impl_ep_field!(coreml, static_input_shapes, bool);
    impl_ep_field!(coreml, subgraph_running, bool);
    impl_ep_field!(coreml, model_format, u8);
    impl_ep_field!(coreml, compute_units, u8);
    impl_ep_field!(coreml, specialization_strategy, u8);

    // ---------------- Nnapi -----------------------
    impl_ep_field!(nnapi, cpu_only, bool);
    impl_ep_field!(nnapi, disable_cpu, bool);
    impl_ep_field!(nnapi, fp16, bool);
    impl_ep_field!(nnapi, nchw, bool);

    // ---------------- ARM NN -----------------------
    impl_ep_field!(armnn, arena_allocator, bool);

    // ---------------- WebGPU -----------------------
    // WebGPU has no configurable parameters currently

    // ---------------- OpenVINO -----------------------
    impl_ep_field!(openvino, dynamic_shapes, bool);
    impl_ep_field!(openvino, opencl_throttling, bool);
    impl_ep_field!(openvino, qdq_optimizer, bool);
    impl_ep_field!(openvino, num_threads, usize);

    // ---------------- OneDNN -----------------------
    impl_ep_field!(onednn, arena_allocator, bool);

    // ---------------- CANN -----------------------
    impl_ep_field!(cann, graph_inference, bool);
    impl_ep_field!(cann, dump_graphs, bool);
    impl_ep_field!(cann, dump_om_model, bool);

    // ---------------- MIGraphX -----------------------
    impl_ep_field!(migraphx, fp16, bool);
    impl_ep_field!(migraphx, exhaustive_tune, bool);
}
