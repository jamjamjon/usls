use std::fmt;

/// Module identifier for different model components.
///
/// Used as keys in `Config::modules` HashMap to identify different model modules.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Module {
    // Vision models (single module)
    Model,

    // Vision-Language models
    Visual,
    Textual,

    // Encoder-Decoder architectures
    Encoder,
    Decoder,
    VisualEncoder,
    TextualEncoder,
    VisualDecoder,
    TextualDecoder,
    TextualDecoderMerged,

    // Specialized components
    SizeEncoder,
    SizeDecoder,
    CoordEncoder,
    CoordDecoder,
    VisualProjection,
    TextualProjection,

    // Custom module (for extensibility)
    Custom(String),
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model => write!(f, "model"),
            Self::Visual => write!(f, "visual"),
            Self::Textual => write!(f, "textual"),
            Self::Encoder => write!(f, "encoder"),
            Self::Decoder => write!(f, "decoder"),
            Self::VisualEncoder => write!(f, "visual_encoder"),
            Self::TextualEncoder => write!(f, "textual_encoder"),
            Self::VisualDecoder => write!(f, "visual_decoder"),
            Self::TextualDecoder => write!(f, "textual_decoder"),
            Self::TextualDecoderMerged => write!(f, "textual_decoder_merged"),
            Self::SizeEncoder => write!(f, "size_encoder"),
            Self::SizeDecoder => write!(f, "size_decoder"),
            Self::CoordEncoder => write!(f, "coord_encoder"),
            Self::CoordDecoder => write!(f, "coord_decoder"),
            Self::VisualProjection => write!(f, "visual_projection"),
            Self::TextualProjection => write!(f, "textual_projection"),
            Self::Custom(name) => write!(f, "custom({})", name),
        }
    }
}
