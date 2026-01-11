use crate::ResizeAlg;

/// Resize mode type enum for convenient mode selection without parameters.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeModeType {
    #[default]
    FitExact,
    FitWidth,
    FitHeight,
    FitAdaptive,
    Letterbox,
}

/// Image resize modes for different scaling strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum ResizeMode {
    /// Stretch to exact dimensions (distorts aspect ratio).
    FitExact {
        width: u32,
        height: u32,
        alg: ResizeAlg,
    },
    /// Fit to width, height scales proportionally.
    FitWidth {
        width: u32,
        height: u32,
        alg: ResizeAlg,
        padding_value: u8,
    },
    /// Fit to height, width scales proportionally.
    FitHeight {
        width: u32,
        height: u32,
        alg: ResizeAlg,
        padding_value: u8,
    },
    /// Fit adaptively (maintain aspect ratio, pad remainder).
    FitAdaptive {
        width: u32,
        height: u32,
        alg: ResizeAlg,
        padding_value: u8,
    },
    /// Letterbox (maintain aspect ratio with centered padding).
    Letterbox {
        width: u32,
        height: u32,
        alg: ResizeAlg,
        padding_value: u8,
    },
}

impl Default for ResizeMode {
    fn default() -> Self {
        Self::FitExact {
            width: 640,
            height: 640,
            alg: ResizeAlg::default(),
        }
    }
}

impl ResizeMode {
    /// Get target width.
    pub fn width(&self) -> u32 {
        match self {
            Self::FitExact { width, .. }
            | Self::FitWidth { width, .. }
            | Self::FitHeight { width, .. }
            | Self::FitAdaptive { width, .. }
            | Self::Letterbox { width, .. } => *width,
        }
    }

    /// Get target height.
    pub fn height(&self) -> u32 {
        match self {
            Self::FitExact { height, .. }
            | Self::FitWidth { height, .. }
            | Self::FitHeight { height, .. }
            | Self::FitAdaptive { height, .. }
            | Self::Letterbox { height, .. } => *height,
        }
    }

    /// Get resize algorithm.
    pub fn alg(&self) -> ResizeAlg {
        match self {
            Self::FitExact { alg, .. }
            | Self::FitWidth { alg, .. }
            | Self::FitHeight { alg, .. }
            | Self::FitAdaptive { alg, .. }
            | Self::Letterbox { alg, .. } => *alg,
        }
    }

    /// Get padding value.
    pub fn padding_value(&self) -> u8 {
        match self {
            Self::FitWidth { padding_value, .. }
            | Self::FitHeight { padding_value, .. }
            | Self::FitAdaptive { padding_value, .. }
            | Self::Letterbox { padding_value, .. } => *padding_value,
            _ => 114, // default for FitExact which doesn't use padding
        }
    }
}
