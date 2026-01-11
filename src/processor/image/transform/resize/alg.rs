//! Resize algorithm types.

use crate::ResizeFilter;

/// Resize algorithm types.
///
/// Mirrors `fast_image_resize::ResizeAlg` for compatibility.
/// CUDA backend currently only supports `Convolution`; CPU supports all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResizeAlg {
    /// Nearest neighbor interpolation.
    /// Fastest but lowest quality.
    Nearest,
    /// Convolution-based resize with adaptive kernel size.
    /// Best quality, used by CUDA backend.
    Convolution(ResizeFilter),
    /// Like Convolution but with fixed kernel size.
    /// Similar to OpenCV behavior.
    Interpolation(ResizeFilter),
    /// Super-sampling with the given multiplier (2-16).
    /// High quality for downscaling.
    SuperSampling(ResizeFilter, u8),
}

impl Default for ResizeAlg {
    fn default() -> Self {
        Self::Convolution(ResizeFilter::default())
    }
}

impl std::fmt::Display for ResizeAlg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nearest => write!(f, "Nearest"),
            Self::Convolution(filter) => write!(f, "Convolution({})", filter),
            Self::Interpolation(filter) => write!(f, "Interpolation({})", filter),
            Self::SuperSampling(filter, m) => write!(f, "SuperSampling({}, {})", filter, m),
        }
    }
}

impl ResizeAlg {
    /// Create a convolution algorithm with the given filter.
    pub fn convolution(filter: ResizeFilter) -> Self {
        Self::Convolution(filter)
    }

    /// Create an interpolation algorithm with the given filter.
    pub fn interpolation(filter: ResizeFilter) -> Self {
        Self::Interpolation(filter)
    }

    /// Create a super-sampling algorithm with the given filter and multiplier.
    /// Multiplier is clamped to 2-16.
    pub fn super_sampling(filter: ResizeFilter, multiplier: u8) -> Self {
        Self::SuperSampling(filter, multiplier.clamp(2, 16))
    }

    pub fn filter(&self) -> Option<ResizeFilter> {
        match self {
            Self::Nearest => None,
            Self::Convolution(f) | Self::Interpolation(f) | Self::SuperSampling(f, _) => Some(*f),
        }
    }

    pub fn build_fir_resizer_and_options(
        &self,
    ) -> anyhow::Result<(fast_image_resize::Resizer, fast_image_resize::ResizeOptions)> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new();

        let alg = match &self {
            Self::Nearest => fast_image_resize::ResizeAlg::Nearest,
            Self::Convolution(filter) => {
                fast_image_resize::ResizeAlg::Convolution((*filter).into())
            }
            Self::Interpolation(filter) => {
                fast_image_resize::ResizeAlg::Interpolation((*filter).into())
            }
            Self::SuperSampling(filter, multiplier) => {
                fast_image_resize::ResizeAlg::SuperSampling((*filter).into(), *multiplier)
            }
        };
        Ok((resizer, options.resize_alg(alg)))
    }
}
