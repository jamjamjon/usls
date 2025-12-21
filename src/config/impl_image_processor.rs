impl crate::Config {
    /// Set processor device.
    pub fn with_image_processor_device(mut self, device: crate::Device) -> Self {
        self.image_processor.device = device;
        self
    }

    /// Set image mean values for normalization.
    pub fn with_image_mean(mut self, mean: [f32; 3]) -> Self {
        self.image_processor.image_mean = Some(mean);
        self
    }

    /// Set image standard deviation values for normalization.
    pub fn with_image_std(mut self, std: [f32; 3]) -> Self {
        self.image_processor.image_std = Some(std);
        self
    }

    /// Set resize mode.
    pub fn with_resize_mode(mut self, mode: crate::ResizeMode) -> Self {
        self.image_processor.resize_mode = mode;
        self
    }

    /// Set resize filter.
    pub fn with_resize_filter(mut self, filter: crate::ResizeFilter) -> Self {
        self.image_processor = self.image_processor.with_resize_filter(filter);
        self
    }

    /// Set whether to normalize images.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.image_processor.normalize = normalize;
        self
    }

    /// Set padding value.
    pub fn with_padding_value(mut self, value: u8) -> Self {
        self.image_processor.padding_value = value;
        self
    }

    /// Set image width.
    pub fn with_image_width(mut self, width: u32) -> Self {
        self.image_processor.image_width = width;
        self
    }

    /// Set image height.
    pub fn with_image_height(mut self, height: u32) -> Self {
        self.image_processor.image_height = height;
        self
    }

    /// Set whether to do resize.
    pub fn with_do_resize(mut self, do_resize: bool) -> Self {
        self.image_processor.do_resize = do_resize;
        self
    }

    /// Set unsigned mode.
    pub fn with_unsigned(mut self, unsigned: bool) -> Self {
        self.image_processor.unsigned = unsigned;
        self
    }

    /// Set up scale.
    pub fn with_up_scale(mut self, up_scale: f32) -> Self {
        self.image_processor.up_scale = up_scale;
        self
    }

    // ===== Super-resolution convenience methods =====

    /// Set pad image flag for super-resolution.
    pub fn with_pad_image(mut self, pad: bool) -> Self {
        self.image_processor.pad_image = pad;
        self
    }

    /// Set pad size for super-resolution.
    pub fn with_pad_size(mut self, size: usize) -> Self {
        self.image_processor.pad_size = size;
        self
    }
}
