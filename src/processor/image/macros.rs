#[allow(unused_macros)]
macro_rules! impl_image_processor_config_methods {
    ($ty:ty, $field:ident) => {
        impl $ty {
            pub fn with_image_width(mut self, image_width: u32) -> Self {
                self.$field = self.$field.with_image_width(image_width);
                self
            }
            pub fn with_image_height(mut self, image_height: u32) -> Self {
                self.$field = self.$field.with_image_height(image_height);
                self
            }
            pub fn with_do_resize(mut self, do_resize: bool) -> Self {
                self.$field = self.$field.with_do_resize(do_resize);
                self
            }
            pub fn with_resize_mode_type(
                mut self,
                resize_mode_type: $crate::ResizeModeType,
            ) -> Self {
                self.$field = self.$field.with_resize_mode_type(resize_mode_type);
                self
            }
            pub fn with_resize_filter(mut self, filter: $crate::ResizeFilter) -> Self {
                self.$field = self.$field.with_resize_filter(filter);
                self
            }
            pub fn with_resize_alg(mut self, resize_alg: $crate::ResizeAlg) -> Self {
                self.$field = self.$field.with_resize_alg(resize_alg);
                self
            }
            pub fn with_padding_value(mut self, padding_value: u8) -> Self {
                self.$field = self.$field.with_padding_value(padding_value);
                self
            }
            pub fn with_normalize(mut self, normalize: bool) -> Self {
                self.$field = self.$field.with_normalize(normalize);
                self
            }
            pub fn with_image_std(mut self, image_std: &[f32; 3]) -> Self {
                self.$field = self.$field.with_image_std(*image_std);
                self
            }
            pub fn with_image_mean(mut self, image_mean: &[f32; 3]) -> Self {
                self.$field = self.$field.with_image_mean(*image_mean);
                self
            }
            pub fn with_unsigned(mut self, unsigned: bool) -> Self {
                self.$field = self.$field.with_unsigned(unsigned);
                self
            }
            pub fn with_pad_image(mut self, pad_image: bool) -> Self {
                self.$field = self.$field.with_pad_image(pad_image);
                self
            }
            pub fn with_pad_size(mut self, pad_size: usize) -> Self {
                self.$field = self.$field.with_pad_size(pad_size);
                self
            }
            pub fn with_image_tensor_layout(
                mut self,
                image_tensor_layout: $crate::ImageTensorLayout,
            ) -> Self {
                self.$field = self.$field.with_image_tensor_layout(image_tensor_layout);
                self
            }
        }
    };
}
