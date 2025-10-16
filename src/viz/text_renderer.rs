use ab_glyph::{FontArc, PxScale};
use aksr::Builder;
use anyhow::Result;
use image::{Rgba, RgbaImage};
use std::sync::OnceLock;

use crate::{Color, Hub};

/// Text rendering engine with font management and styling capabilities.
#[derive(Builder, Clone, Debug)]
pub struct TextRenderer {
    #[args(except(setter))]
    font: Option<FontArc>,
    font_size: f32,
    _scale: f32,
}

// Global font cache using OnceLock for lazy initialization
static DEFAULT_FONT: OnceLock<FontArc> = OnceLock::new();

impl Default for TextRenderer {
    fn default() -> Self {
        Self {
            font: None,
            font_size: 24.0,
            _scale: 6.666667,
        }
    }
}

impl TextRenderer {
    /// Get the font, loading it lazily if needed
    fn get_font(&self) -> Result<&FontArc> {
        // First try to get user-specified font
        if let Some(ref font) = self.font {
            return Ok(font);
        }

        // If no user font, try to get from global cache
        if let Some(font) = DEFAULT_FONT.get() {
            return Ok(font);
        }

        // Load default font and cache it globally
        let font = Self::load_font(None).or_else(|err| {
            log::info!("Failed to load online font: {err}, try using system font.");
            Self::create_fallback_font()
        })?;

        DEFAULT_FONT
            .set(font)
            .map_err(|_| anyhow::anyhow!("Failed to cache font"))?;
        Ok(DEFAULT_FONT.get().unwrap())
    }

    /// Create a fallback font when default font loading fails
    fn create_fallback_font() -> Result<FontArc> {
        // Try to load a preferred system font first
        let system_font_paths = [
            // macOS fonts
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            // Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            // Windows fonts
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
        ];
        for path in &system_font_paths {
            if let Ok(font_data) = std::fs::read(path) {
                if let Ok(font) = FontArc::try_from_vec(font_data) {
                    log::info!("Using system font: {}", path);
                    return Ok(font);
                }
            }
        }
        log::info!("No preferred system fonts available, try to scan all system fonts");

        // Get system font directories for the current platform
        // - macOS: `/System/Library/Fonts/`
        // - Linux: `/usr/share/fonts/truetype/`
        // - Windows: `C:\Windows\Fonts\`, `C:\Program Files\Common Files\Microsoft Shared\Fonts\`
        let font_directory = {
            #[cfg(target_os = "macos")]
            {
                "/System/Library/Fonts/"
            }
            #[cfg(target_os = "linux")]
            {
                "/usr/share/fonts/truetype/"
            }
            #[cfg(target_os = "windows")]
            {
                "C:\\Windows\\Fonts\\"
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
            {
                anyhow::bail!("Unsupported platform: font scanning not available")
            }
        };
        log::debug!("Scanning system font directories: {:?}", font_directory);

        // Scan all system fonts: TTF, OTF, TTC
        let mut font_paths = Vec::new();
        Self::collect_font_paths_recursive(font_directory, &mut font_paths);

        // Then try to load fonts from collected paths
        for font_path in font_paths {
            if let Ok(font_data) = std::fs::read(&font_path) {
                if let Ok(font) = FontArc::try_from_vec(font_data) {
                    log::info!("Successfully loaded system font: {}", font_path.display());
                    return Ok(font);
                }
            }
        }

        anyhow::bail!(
            "No system fonts available. Please use Annotator::default().with_font(\"path/to/font.ttf\") to specify a custom font."
        )
    }

    /// Recursively collect font file paths from a directory
    fn collect_font_paths_recursive(dir_path: &str, font_paths: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir_path) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_dir() {
                    // Recursively scan subdirectories
                    Self::collect_font_paths_recursive(&path.to_string_lossy(), font_paths);
                } else if let Some(extension) = path.extension() {
                    let ext_str = extension.to_string_lossy().to_lowercase();
                    if matches!(ext_str.as_str(), "ttf" | "otf" | "ttc") {
                        font_paths.push(path);
                    }
                }
            }
        }
    }

    /// Load custom font
    fn load_font(path: Option<&str>) -> Result<FontArc> {
        let path_font = match path {
            None => Hub::default().try_fetch("fonts/JetBrainsMono-Regular.ttf")?,
            Some(p) => p.into(),
        };
        let buf = std::fs::read(path_font)?;
        let font = FontArc::try_from_vec(buf)?;

        Ok(font)
    }

    pub fn with_font(mut self, path: &str) -> Result<Self> {
        self.font = Some(Self::load_font(Some(path))?);
        Ok(self)
    }

    pub fn text_size(&self, text: &str) -> Result<(u32, u32)> {
        let font = self.get_font()?;
        let scale = PxScale::from(self.font_size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, font, text);
        let text_h = text_h + text_h / 3;

        Ok((text_w, text_h))
    }

    pub fn render(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: f32,
        y: f32,
        color: Color,
        background_color: Color,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let font = self.get_font()?;
        let scale = PxScale::from(self.font_size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, font, text);
        let text_h = text_h + text_h / 3;
        let (left, top) = self.calculate_position(x, y, text_w, text_h, img.width());

        imageproc::drawing::draw_filled_rect_mut(
            img,
            imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
            Rgba(background_color.into()),
        );

        imageproc::drawing::draw_text_mut(
            img,
            Rgba(color.into()),
            left,
            top + self.calculate_text_offset(),
            scale,
            font,
            text,
        );

        Ok(())
    }

    fn calculate_position(
        &self,
        x: f32,
        y: f32,
        text_w: u32,
        text_h: u32,
        img_width: u32,
    ) -> (i32, i32) {
        let top = if y > text_h as f32 {
            (y.round() as u32 - text_h) as i32
        } else {
            0
        };

        let mut left = x as i32;
        if left + text_w as i32 > img_width as i32 {
            left = img_width as i32 - text_w as i32;
        }

        (left, top)
    }

    fn calculate_text_offset(&self) -> i32 {
        -(self.font_size / self._scale).floor() as i32 + 1
    }
}
