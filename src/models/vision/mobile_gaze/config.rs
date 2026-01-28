///
/// > # MobileGaze: Pre-trained mobile nets for Gaze-Estimation
/// >
/// # Paper & Code
///
/// > - **GitHub**: [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation)
/// >
/// # Model Variants
///
/// > - **ResNet-18**: Balanced accuracy and speed (11.7M parameters)
/// > - **ResNet-34**: Higher accuracy with moderate speed (21.8M parameters)
/// > - **ResNet-50**: Best accuracy with slower speed (25.6M parameters)
/// > - **MobileNet-V2**: Fastest inference for mobile devices (3.5M parameters)
/// > - **MobileOne-S0**: Ultra-fast with reparameterization (4.1M parameters)
///
/// Model configuration for `MobileGaze`
///
impl crate::Config {
    /// Base configuration for MobileGaze models
    ///
    /// Sets up the standard MobileGaze configuration with:
    /// - Input size: 448x448
    /// - 3-channel RGB input
    /// - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    /// - Bilinear interpolation for resizing
    /// - FitExact resize mode for consistent input dimensions
    pub fn mobile_gaze() -> Self {
        Self::default()
            .with_name("mobile_gaze")
            .with_model_ixx(0, 0, (1, 1, 4))
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 448)
            .with_model_ixx(0, 3, 448)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_resize_alg(crate::ResizeAlg::Interpolation(
                crate::ResizeFilter::Bilinear,
            ))
            .with_resize_mode_type(crate::ResizeModeType::FitExact)
            .with_normalize(true)
    }

    /// ResNet-18 backbone configuration
    ///
    /// Uses ResNet-18 as the feature extractor, providing a good balance
    /// between accuracy and inference speed for real-time applications.
    pub fn mobile_gaze_r18() -> Self {
        Self::mobile_gaze().with_model_file("resnet18.onnx")
    }

    /// ResNet-34 backbone configuration
    ///
    /// Uses ResNet-34 for improved accuracy while maintaining reasonable
    /// inference speed for most real-time use cases.
    pub fn mobile_gaze_r34() -> Self {
        Self::mobile_gaze().with_model_file("resnet34.onnx")
    }

    /// ResNet-50 backbone configuration
    ///
    /// Uses ResNet-50 for the highest accuracy among ResNet variants,
    /// suitable for applications where accuracy is prioritized over speed.
    pub fn mobile_gaze_r50() -> Self {
        Self::mobile_gaze().with_model_file("resnet50.onnx")
    }

    /// MobileNet-V2 backbone configuration
    ///
    /// Uses MobileNet-V2 for ultra-fast inference on mobile and edge devices,
    /// ideal for battery-powered or resource-constrained applications.
    pub fn mobile_gaze_mobilenet_v2() -> Self {
        Self::mobile_gaze().with_model_file("mobilenet-v2.onnx")
    }

    /// MobileOne-S0 backbone configuration
    ///
    /// Uses MobileOne-S0 with structural reparameterization for the fastest
    /// inference speed while maintaining good accuracy for real-time gaze tracking.
    pub fn mobile_gaze_mobileone_s0() -> Self {
        Self::mobile_gaze().with_model_file("mobileone-s0.onnx")
    }
}
