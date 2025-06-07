use std::str::FromStr;

#[derive(Debug, Clone, Ord, Eq, PartialOrd, PartialEq)]
pub enum Task {
    /// Image classification task.
    /// Input: image
    /// Output: a label representing the class of the image
    ImageClassification,

    /// Multi-label image tagging task.
    /// Input: image
    /// Output: multiple labels representing different categories in the image
    ImageTagging,

    /// Image captioning task, generating descriptions with different levels of detail.
    /// Input: image
    /// Output: a text description, `u8` represents the level of detail:
    /// 0 for brief, 1 for detailed, 2 for more detailed
    Caption(u8),

    /// Region proposal task, detecting all objects in the image.
    /// Input: image
    /// Output: bounding boxes (bboxes)
    RegionProposal,

    /// Object detection task, detecting all objects in the image.
    /// Input: image
    /// Output: bounding boxes (bboxes), class labels, and optional scores for the detected objects
    ObjectDetection,
    OrientedObjectDetection,
    Obb,

    /// Open set detection task, detecting and classifying objects in an image, with the ability to handle unseen or unknown objects.
    /// Input: image
    /// Output: bounding boxes, class labels (including an "unknown" category for unfamiliar objects), and detection scores
    /// Open set detection task, with String query
    OpenSetDetection(String),
    /// Task for generating brief descriptions of dense regions in the image.
    /// Input: image
    /// Output: bounding boxes (bboxes), brief phrase labels, and optional scores for detected regions
    DenseRegionCaption,

    /// Keypoint detection task, detecting keypoints in an image.
    /// This can include human body parts (e.g., hands, feet, joints) or other objects.
    /// Input: image
    /// Output: coordinates of detected keypoints
    KeypointsDetection,
    Pose,
    OpenSetKeypointsDetection(String),

    /// Semantic segmentation task, segmenting the image into different semantic regions.
    /// Input: image
    /// Output: per-pixel class labels indicating object or background
    SemanticSegmentation,

    ImageFeatureExtraction,
    TextFeatureExtraction,

    /// Instance segmentation task, detecting and segmenting individual object instances.
    /// Input: image
    /// Output: pixel masks for each object instance
    InstanceSegmentation,

    /// Depth estimation task, predicting the distance of each pixel from the camera.
    /// Input: image
    /// Output: a depth map where each pixel has a depth value
    DepthEstimation,

    /// Surface normal prediction task, predicting the surface normal vector for each pixel.
    /// Input: image
    /// Output: a normal map where each pixel has a surface normal vector
    SurfaceNormalPrediction,

    /// Image-to-image generation task, transforming one image into another.
    /// Input: image
    /// Output: a generated image
    ImageToImageGeneration,

    /// Text-to-image generation task, generating an image based on a text description.
    /// Input: text
    /// Output: a generated image
    TextToImageGeneration,

    /// Inpainting task, filling in missing or corrupted parts of an image.
    /// Input: image with missing or corrupted regions
    /// Output: a complete image with the missing parts filled in
    Inpainting,

    /// Super-resolution task, enhancing the resolution of an image.
    /// Input: low-resolution image
    /// Output: high-resolution image
    SuperResolution,

    /// Image denoising task, removing noise from an image.
    /// Input: noisy image
    /// Output: denoised image
    Denoising,

    /// Phrase grounding task, finding the region in an image corresponding to a text description.
    /// Input: image and text
    /// Output: image region and the corresponding phrase
    /// caption to phrase grounding
    CaptionToPhraseGrounding(String),

    /// Referring expression segmentation task, segmenting objects in the image based on a text description.
    /// Input: image and referring expression
    /// Output: a segmentation mask for the object referred to by the text
    ReferringExpressionSegmentation(String),

    /// Region-to-segmentation task, similar to combining object detection with segmentation (e.g., YOLO + SAM).
    /// Input: image and region proposals
    /// Output: segmentation masks for the regions
    /// Region, bbox: top-left, bottom-right
    RegionToSegmentation(usize, usize, usize, usize),

    /// Region-to-category classification task, classifying the object in a given region of the image.
    /// Input: image and region
    /// Output: class label for the region
    /// Region, bbox: top-left, bottom-right
    RegionToCategory(usize, usize, usize, usize),

    /// Region-to-description task, generating a detailed description for a given region in the image.
    /// Input: image and region
    /// Output: a detailed textual description for the region
    /// Region, bbox: top-left, bottom-right
    RegionToDescription(usize, usize, usize, usize),

    /// Visual question answering (VQA) task, answering questions related to an image.
    /// Input: image and question text
    /// Output: the answer to the question
    Vqa(String),

    /// Optical character recognition (OCR) task, recognizing text in an image.
    /// Input: image
    /// Output: recognized text
    Ocr,

    /// OCR task with region information, recognizing text and returning its location in the image.
    /// Input: image
    /// Output: recognized text and its bounding box in the image
    OcrWithRegion,
}

impl std::fmt::Display for Task {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::ImageClassification => "image-classification",
            Self::ObjectDetection => "object-detection",
            Self::Pose => "pose",
            Self::KeypointsDetection => "pose-detection",
            Self::InstanceSegmentation => "instance-segmentation",
            Self::Obb => "obb",
            Self::OrientedObjectDetection => "oriented-object-detection",
            Self::DepthEstimation => "depth",
            Self::Caption(0) => "caption",
            Self::Caption(1) => "detailed-caption",
            Self::Caption(2) => "more-detailed-caption",
            Self::ImageTagging => "image-tagging",
            Self::Ocr => "ocr",
            Self::OcrWithRegion => "ocr-with-region",
            Self::Vqa(_) => "vqa",
            Self::OpenSetKeypointsDetection(_) => "open-set-keypoints-detection",
            _ => todo!(),
        };
        write!(f, "{}", x)
    }
}

impl FromStr for Task {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO
        match s.to_lowercase().as_str() {
            "cls" | "classify" | "classification" => Ok(Self::ImageClassification),
            "det" | "od" | "detect" => Ok(Self::ObjectDetection),
            "kpt" | "pose" => Ok(Self::KeypointsDetection),
            "seg" | "segment" => Ok(Self::InstanceSegmentation),
            "obb" => Ok(Self::OrientedObjectDetection),
            "cap" | "cap0" | "caption" => Ok(Self::Caption(0)),
            "cap1" | "caption1" => Ok(Self::Caption(1)),
            "cap2" | "caption2" => Ok(Self::Caption(2)),
            x if x.contains(":") => {
                let t_tt: Vec<&str> = x.trim().split(':').collect();
                let (t, tt) = match t_tt.len() {
                    2 => (t_tt[0].trim(), t_tt[1].trim()),
                    _ => anyhow::bail!(
                        "Fail to parse task: {x}. Expect: `task:content`. e.g. `vqa:What's in this image?`"
                    ),
                };
                match t {
                    "cap" | "caption" => Ok(Self::Caption(tt.parse::<usize>().unwrap_or(0) as u8)),
                    "vqa" => Ok(Self::Vqa(tt.into())),
                    "open-det" | "open-od" => Ok(Self::OpenSetDetection(tt.into())),
                    "open-kpt" | "open-pose" => Ok(Self::OpenSetKeypointsDetection(tt.into())),
                    _ => todo!(),
                }
            }
            _ => todo!(),
        }
    }
}

impl Task {
    pub fn yolo_str(&self) -> &'static str {
        match self {
            Self::ImageClassification => "cls",
            Self::ObjectDetection => "det",
            Self::Pose | Self::KeypointsDetection => "pose",
            Self::InstanceSegmentation => "seg",
            Self::Obb | Self::OrientedObjectDetection => "obb",
            x => unimplemented!("Unsupported YOLO Task: {}", x),
        }
    }

    pub fn prompt_for_florence2(&self) -> anyhow::Result<String> {
        let prompt = match self {
            Self::Caption(0) => "What does the image describe?".to_string(),
            Self::Caption(1) => "Describe in detail what is shown in the image.".to_string(),
            Self::Caption(2) => "Describe with a paragraph what is shown in the image.".to_string(),
            Self::Ocr => "What is the text in the image?".to_string(),
            Self::OcrWithRegion => "What is the text in the image, with regions?".to_string(),
            Self::ObjectDetection => {
                "Locate the objects with category name in the image.".to_string()
            }
            Self::DenseRegionCaption => {
                "Locate the objects in the image, with their descriptions.".to_string()
            }
            Self::RegionProposal => "Locate the region proposals in the image.".to_string(),
            Self::OpenSetDetection(text) => {
                format!("Locate {} in the image.", text)
            }
            Self::CaptionToPhraseGrounding(text) => {
                format!("Locate the phrases in the caption: {}", text)
            }
            Self::ReferringExpressionSegmentation(text) => {
                format!("Locate {} in the image with mask", text)
            }
            Self::RegionToSegmentation(x0, y0, x1, y1) => {
                format!(
                    "What is the polygon mask of region <loc_{}><loc_{}><loc_{}><loc_{}>",
                    x0, y0, x1, y1
                )
            }
            Self::RegionToCategory(x0, y0, x1, y1) => {
                format!(
                    "What is the region <loc_{}><loc_{}><loc_{}><loc_{}>?",
                    x0, y0, x1, y1
                )
            }
            Self::RegionToDescription(x0, y0, x1, y1) => {
                format!(
                    "What does the region <loc_{}><loc_{}><loc_{}><loc_{}> describe?",
                    x0, y0, x1, y1
                )
            }
            x => anyhow::bail!("Unsupported Florence2 task: {:?}", x),
        };

        Ok(prompt)
    }
}
