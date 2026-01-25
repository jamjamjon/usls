mod config;
mod r#impl;
mod prompt;

pub use prompt::YOLOEPrompt;
pub use r#impl::YOLOEPromptBased;
pub type YOLOE = YOLOEPromptBased;
