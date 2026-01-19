///
/// > # BiRefNet: Bimodal Referencing Network for Precise Image Segmentation
/// >
/// > Advanced image segmentation model that achieves high-precision segmentation through bimodal referencing mechanisms.
/// > The model excels in various segmentation tasks including salient object detection, camouflaged object detection, and portrait matting.
/// >
/// > # Paper & Code
/// >
/// > - **GitHub**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
/// > - **Paper**: [BiRefNet: Bimodal Referencing Network for Precise Image Segmentation](https://arxiv.org/abs/2401.11691)
/// >
/// > # Model Variants
/// >
/// > - **birefnet-general**: General-purpose model for common segmentation tasks
/// > - **birefnet-general-bb_swin_v1_tiny**: General-purpose model with Swin-V1-Tiny backbone for balanced performance and efficiency
/// > - **birefnet-lite-general-2k**: Lightweight general model for 2K resolution
/// > - **birefnet-hr-general**: High-resolution general segmentation model
/// > - **birefnet-cod**: Specialized for Camouflaged Object Detection (COD)
/// > - **birefnet-dis**: Optimized for Dichotomous Image Segmentation (DIS)
/// > - **birefnet-hrsod-dhu**: High-Resolution Salient Object Detection variant
/// > - **birefnet-massive**: Massive model trained on multiple datasets including DIS5K and TE datasets for robust performance
/// > - **birefnet-matting**: Portrait matting and background removal
/// > - **birefnet-hr-matting**: High-Resolution Portrait matting and background removal
/// > - **birefnet-portrait**: Specialized for portrait segmentation
/// >
/// > # Implemented Features / Tasks
/// >
/// > - [X] **Dichotomous Image Segmentation**: Basic foreground/background separation
/// > - [X] **Camouflaged Object Detection**: Segment camouflaged objects in complex scenes
/// > - [X] **High-Resolution Salient Object Detection**: Detect salient objects in high-res images
/// > - [X] **Portrait Matting**: High-quality portrait background removal
/// > - [X] **General Segmentation**: Versatile model for various segmentation tasks
/// >
/// Model configuration for `BiRefNet`
///
impl crate::Config {
    /// Base configuration for BiRefNet models with standard normalization and image preprocessing
    pub fn birefnet() -> Self {
        Self::default()
            .with_name("birefnet")
            .with_model_ixx(0, 0, 1)
            .with_model_ixx(0, 1, 3)
            .with_model_ixx(0, 2, 1024)
            .with_model_ixx(0, 3, 1024)
            .with_image_mean([0.485, 0.456, 0.406])
            .with_image_std([0.229, 0.224, 0.225])
            .with_normalize(true)
    }

    /// Camouflaged Object Detection (COD) model for segmenting objects that blend with their surroundings
    pub fn birefnet_cod() -> Self {
        Self::birefnet().with_model_file("birefnet/COD-epoch-125.onnx")
    }

    /// Dichotomous Image Segmentation (DIS) model for basic foreground/background separation
    pub fn birefnet_dis() -> Self {
        Self::birefnet().with_model_file("birefnet/DIS-epoch-590.onnx")
    }

    /// High-Resolution Salient Object Detection (HRSOD) model trained on DHU dataset
    pub fn birefnet_hrsod_dhu() -> Self {
        Self::birefnet().with_model_file("birefnet/HRSOD_DHU-epoch-115.onnx")
    }

    /// Massive model trained on multiple datasets including DIS5K and TE datasets for robust performance
    pub fn birefnet_massive() -> Self {
        Self::birefnet().with_model_file("birefnet/massive-TR_DIS5K_TR_TEs-epoch-420.onnx")
    }

    /// General-purpose model with Swin-V1-Tiny backbone for balanced performance and efficiency
    pub fn birefnet_general_bb_swin_v1_tiny() -> Self {
        Self::birefnet().with_model_file("birefnet/general-bb_swin_v1_tiny-epoch-232.onnx")
    }

    /// General-purpose model for versatile segmentation tasks across different domains
    pub fn birefnet_general() -> Self {
        Self::birefnet().with_model_file("birefnet/general-epoch-244.onnx")
    }

    /// High-resolution general segmentation model for processing larger images with fine details
    pub fn birefnet_hr_general() -> Self {
        Self::birefnet().with_model_file("birefnet/HR-general-epoch-130.onnx")
    }

    /// Lightweight general model optimized for 2K resolution images with efficient inference
    pub fn birefnet_lite_general_2k() -> Self {
        Self::birefnet().with_model_file("birefnet/lite-general-2K-epoch-232.onnx")
    }

    /// Specialized portrait segmentation model for high-quality portrait background removal
    pub fn birefnet_portrait() -> Self {
        Self::birefnet().with_model_file("birefnet/portrait-epoch-150.onnx")
    }

    /// Portrait matting model for precise hair and fine detail preservation in portraits
    pub fn birefnet_matting() -> Self {
        Self::birefnet().with_model_file("birefnet/matting-epoch-100.onnx")
    }

    /// High-resolution portrait matting model for detailed portrait segmentation on larger images
    pub fn birefnet_hr_matting() -> Self {
        Self::birefnet().with_model_file("birefnet/HR-matting-epoch-135.onnx")
    }
}
