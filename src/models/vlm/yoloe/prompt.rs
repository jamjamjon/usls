use crate::Hbb;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct YOLOEPrompt {
    pub texts: Vec<String>,
    pub boxes: Vec<(Hbb, PathBuf)>,
}