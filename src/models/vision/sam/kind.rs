#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamKind {
    /// Original SAM model
    Sam,
    /// SAM 2.0 with hierarchical architecture
    Sam2,
    /// Mobile optimized SAM
    MobileSam,
    /// High quality SAM with better segmentation
    SamHq,
    /// Efficient SAM with edge-based segmentation
    EdgeSam,
}

impl std::str::FromStr for SamKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sam" => Ok(Self::Sam),
            "sam2" => Ok(Self::Sam2),
            "mobile-sam" | "mobile_sam" | "mobilesam" => Ok(Self::MobileSam),
            "sam-hq" | "sam_hq" | "samhq" => Ok(Self::SamHq),
            "edge-sam" | "edge_sam" | "edgesam" => Ok(Self::EdgeSam),
            _ => anyhow::bail!("Unknown SAM kind: {s}"),
        }
    }
}
