/// Image tensor layout formats for organizing image data in memory.
///
/// This enum defines different ways to arrange image pixel data in tensors:
/// - **Batch formats** (with batch dimension): `NCHW`, `NHWC`
/// - **Single image formats** (no batch dimension): `CHW`, `HWC`
///
/// The format affects how image data is stored and accessed in memory,
/// which is important for compatibility with different model architectures
/// (e.g., PyTorch typically uses NCHW, TensorFlow uses NHWC).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTensorLayout {
    /// NCHW format: (batch, channel, height, width)
    /// Channels-first layout, commonly used in PyTorch models.
    #[default]
    NCHW,
    /// NHWC format: (batch, height, width, channel)
    /// Channels-last layout, commonly used in TensorFlow models.
    NHWC,
    /// CHW format: (channel, height, width)
    /// Single image with channels-first layout (no batch dimension).
    CHW,
    /// HWC format: (height, width, channel)
    /// Single image with channels-last layout (no batch dimension).
    HWC,
    // // TODO: multi-view depth estimation!
    // NMHWC, // (batch, num_images, height, width, channel)
    // NMCHW, // (batch, num_images, channel, height, width)
}

impl ImageTensorLayout {
    /// Check if layout is channels-first (CHW or NCHW).
    #[inline]
    pub fn is_channels_first(&self) -> bool {
        matches!(self, Self::CHW | Self::NCHW)
    }

    /// Check if layout is channels-last (HWC or NHWC).
    #[inline]
    pub fn is_channels_last(&self) -> bool {
        matches!(self, Self::HWC | Self::NHWC)
    }
}
