#[derive(Debug, Clone)]
pub enum Task {
    // vision
    ImageClassification,
    ObjectDetection,
    KeypointsDetection,
    RegisonProposal,
    PoseEstimation,
    SemanticSegmentation,
    InstanceSegmentation,
    DepthEstimation,
    SurfaceNormalPrediction,
    Image2ImageGeneration,
    Inpainting,
    SuperResolution,
    Denoising,

    // vl
    Tagging,
    Captioning,
    DetailedCaptioning,
    MoreDetailedCaptioning,
    PhraseGrounding,
    Vqa,
    Ocr,
    Text2ImageGeneration,
}
