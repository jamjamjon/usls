#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Kind {
    // Do we really need this?
    Vision,
    Language,
    VisionLanguage,
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::Vision => "visual",
            Self::Language => "textual",
            Self::VisionLanguage => "vl",
        };
        write!(f, "{}", x)
    }
}
