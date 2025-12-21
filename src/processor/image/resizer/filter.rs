//! Resize filter types for image resizing.

/// Filter types for image resizing.
///
/// Mirrors `fast_image_resize::FilterType` for compatibility while providing
/// a stable, crate-owned enum that can be used across all backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResizeFilter {
    /// Box filter. Each source pixel contributes equally.
    /// Minimal kernel size: 1x1 px.
    Box,
    /// Bilinear interpolation.
    /// Minimal kernel size: 2x2 px.
    #[default]
    Bilinear,
    /// Hamming filter. Similar quality to bicubic for downscaling.
    /// Minimal kernel size: 2x2 px.
    Hamming,
    /// Catmull-Rom bicubic filter.
    /// Minimal kernel size: 4x4 px.
    CatmullRom,
    /// Mitchell-Netravali bicubic filter.
    /// Minimal kernel size: 4x4 px.
    Mitchell,
    /// Gaussian filter (sigma=0.5).
    /// Minimal kernel size: 6x6 px.
    Gaussian,
    /// Lanczos3 filter (truncated sinc).
    /// Minimal kernel size: 6x6 px.
    Lanczos3,
}

// TODO: map to fast_image_resize::FilterType => build_resizer_filter()

impl std::str::FromStr for ResizeFilter {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "box" => Ok(Self::Box),
            "bilinear" => Ok(Self::Bilinear),
            "hamming" => Ok(Self::Hamming),
            "catmullrom" | "catmull_rom" | "catmull-rom" => Ok(Self::CatmullRom),
            "mitchell" => Ok(Self::Mitchell),
            "gaussian" => Ok(Self::Gaussian),
            "lanczos3" | "lanczos" => Ok(Self::Lanczos3),
            x => anyhow::bail!("Unsupported ResizeFilter: {x}"),
        }
    }
}

impl std::fmt::Display for ResizeFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Box => write!(f, "Box"),
            Self::Bilinear => write!(f, "Bilinear"),
            Self::Hamming => write!(f, "Hamming"),
            Self::CatmullRom => write!(f, "CatmullRom"),
            Self::Mitchell => write!(f, "Mitchell"),
            Self::Gaussian => write!(f, "Gaussian"),
            Self::Lanczos3 => write!(f, "Lanczos3"),
        }
    }
}
