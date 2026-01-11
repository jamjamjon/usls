///
/// > # LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
/// >
/// > Efficient semantic segmentation network with minimal parameters for real-time applications.
/// >
/// > # Paper & Code
/// >
/// > - **Paper**: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)
/// >
/// > # Model Variants
/// >
/// > - **linknet-r18**: ResNet-18 backbone for efficient segmentation
/// > - **linknet-r34**: ResNet-34 backbone for balanced performance
/// > - **linknet-r50**: ResNet-50 backbone for high accuracy
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Semantic Segmentation**: Efficient pixel-wise segmentation
/// > - [X] **Real-time Performance**: Only 11.5M parameters and 21.2 GFLOPs
/// >
/// Model configuration for `LinkNet`
///
impl crate::Config {
    /// Base configuration for LinkNet models
    pub fn linknet() -> Self {
        Self::fast()
            .with_name("linknet")
            .with_image_mean([0.798, 0.785, 0.772])
            .with_image_std([0.264, 0.2749, 0.287])
    }

    /// ResNet-18 backbone for efficient segmentation
    pub fn linknet_r18() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r18.onnx")
    }

    /// ResNet-34 backbone for balanced performance
    pub fn linknet_r34() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r34.onnx")
    }

    /// ResNet-50 backbone for high accuracy
    pub fn linknet_r50() -> Self {
        Self::linknet().with_model_file("felixdittrich92-r50.onnx")
    }
}
