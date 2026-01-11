/// Convert multiple tensors to ORT SessionInputs.
///
/// This macro provides a convenient way to convert `X` or `XView` tensors to ORT input format.
/// It supports both positional and named input styles, and allows **mixing** different input types.
///
/// **Note**: This macro does NOT perform dtype alignment. The tensors must already have
/// the correct dtype expected by the model. For automatic dtype alignment, use `&[X]` input.
///
/// # Input Types (can be mixed freely)
/// - `X<A>` (owned) → moves data into ORT tensor
/// - `&X<A>` (borrowed) → clones data into ORT tensor
/// - `XView<'a, A>` (view) → **zero-copy** reference to original data
///
/// # Examples
/// ```ignore
/// // Owned inputs
/// let inputs = inputs![x1, x2]?;
///
/// // Borrowed inputs
/// let inputs = inputs![&x1, &x2]?;
///
/// // Zero-copy views
/// let inputs = inputs![x1.view(), x2.view()]?;
///
/// // Mixed inputs (owned + borrowed + view)
/// let inputs = inputs![x1, &x2, x3.view()]?;
///
/// // By name
/// let inputs = inputs!["images" => x1, "masks" => &x2]?;
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! inputs {
    // Positional inputs
    ($($t:expr),+ $(,)?) => {{
        (|| -> ::anyhow::Result<_> {
            Ok([ $(
                ::ort::session::SessionInputValue::try_from($t)?,
            )+ ])
        })()
    }};
    // Named inputs
    ($($name:expr => $t:expr),+ $(,)?) => {{
        (|| -> ::anyhow::Result<_> {
            Ok(vec![ $(
                (
                    ::std::borrow::Cow::<str>::from($name),
                    ::ort::session::SessionInputValue::try_from($t)?
                ),
            )+ ])
        })()
    }};
}

#[allow(unused_macros)]
macro_rules! impl_ort_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            paste::paste! {
                pub fn [<with_ $field _file>](mut self, file: &str) -> Self {
                    self.$field = self.$field.with_file(file);
                    self
                }
                pub fn [<with_ $field _external_data_file>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_external_data_file(x);
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
                pub fn [<with_ $field _num_dry_run>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_dry_run(x);
                    self
                }
                pub fn [<with_ $field _ixx>](mut self, i: usize, ii: usize, x: $crate::MinOptMax) -> Self {
                    self.$field = self.$field.with_ixx(i, ii, x);
                    self
                }
                // global
                pub fn [<with_ $field _graph_opt_level>](mut self, x: u8) -> Self {
                    self.$field = self.$field.with_graph_opt_level(x);
                    self
                }
                pub fn [<with_ $field _num_intra_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_intra_threads(x);
                    self
                }
                pub fn [<with_ $field _num_inter_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_num_inter_threads(x);
                    self
                }
                // ep configuration methods - delegate to the field's ep methods
                pub fn [<with_ $field _cpu_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cpu_arena_allocator(x);
                    self
                }
                pub fn [<with_ $field _openvino_dynamic_shapes>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_dynamic_shapes(x);
                    self
                }
                pub fn [<with_ $field _openvino_opencl_throttling>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_opencl_throttling(x);
                    self
                }
                pub fn [<with_ $field _openvino_qdq_optimizer>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_openvino_qdq_optimizer(x);
                    self
                }
                pub fn [<with_ $field _openvino_num_threads>](mut self, x: usize) -> Self {
                    self.$field = self.$field.with_openvino_num_threads(x);
                    self
                }
                pub fn [<with_ $field _onednn_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_onednn_arena_allocator(x);
                    self
                }
                pub fn [<with_ $field _tensorrt_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_fp16(x);
                    self
                }
                pub fn [<with_ $field _tensorrt_engine_cache>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_engine_cache(x);
                    self
                }
                pub fn [<with_ $field _tensorrt_timing_cache>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_tensorrt_timing_cache(x);
                    self
                }
                pub fn [<with_ $field _coreml_static_input_shapes>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_coreml_static_input_shapes(x);
                    self
                }
                pub fn [<with_ $field _coreml_subgraph_running>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_coreml_subgraph_running(x);
                    self
                }
                pub fn [<with_ $field _cann_graph_inference>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_graph_inference(x);
                    self
                }
                pub fn [<with_ $field _cann_dump_graphs>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_dump_graphs(x);
                    self
                }
                pub fn [<with_ $field _cann_dump_om_model>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_cann_dump_om_model(x);
                    self
                }
                pub fn [<with_ $field _nnapi_cpu_only>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_cpu_only(x);
                    self
                }
                pub fn [<with_ $field _nnapi_disable_cpu>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_disable_cpu(x);
                    self
                }
                pub fn [<with_ $field _nnapi_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_fp16(x);
                    self
                }
                pub fn [<with_ $field _nnapi_nchw>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_nnapi_nchw(x);
                    self
                }
                pub fn [<with_ $field _armnn_arena_allocator>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_armnn_arena_allocator(x);
                    self
                }
                pub fn [<with_ $field _migraphx_fp16>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_migraphx_fp16(x);
                    self
                }
                pub fn [<with_ $field _migraphx_exhaustive_tune>](mut self, x: bool) -> Self {
                    self.$field = self.$field.with_migraphx_exhaustive_tune(x);
                    self
                }
            }
        }
    };
}
