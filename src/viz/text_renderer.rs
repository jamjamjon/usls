use ab_glyph::{FontArc, PxScale};
use aksr::Builder;
use anyhow::Result;
use image::{Rgba, RgbaImage};
use std::sync::OnceLock;

use crate::{Color, Hub, TextStyleMode};

/// Text rendering engine with font management and styling capabilities.
///
/// The font_size field is the global default. Individual TextStyle can override
/// this with their own font_size setting.
#[derive(Builder, Clone, Debug)]
pub struct TextRenderer {
    #[args(except(setter))]
    font: Option<FontArc>,
    /// Default font size (used when TextStyle.font_size is None)
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
            tracing::info!("Failed to load online font: {err}, try using system font.");
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
                    tracing::info!("Using system font: {}", path);
                    return Ok(font);
                }
            }
        }
        tracing::info!("No preferred system fonts available, try to scan all system fonts");

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
        tracing::debug!("Scanning system font directories: {:?}", font_directory);

        // Scan all system fonts: TTF, OTF, TTC
        let mut font_paths = Vec::new();
        Self::collect_font_paths_recursive(font_directory, &mut font_paths);

        // Then try to load fonts from collected paths
        for font_path in font_paths {
            if let Ok(font_data) = std::fs::read(&font_path) {
                if let Ok(font) = FontArc::try_from_vec(font_data) {
                    tracing::info!("Successfully loaded system font: {}", font_path.display());
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

    /// Get the effective font size (uses provided override or falls back to default)
    pub fn effective_font_size(&self, override_size: Option<f32>) -> f32 {
        override_size.unwrap_or(self.font_size)
    }

    /// Calculate text dimensions with optional font size override
    pub fn text_size_with(&self, text: &str, font_size: Option<f32>) -> Result<(u32, u32)> {
        let font = self.get_font()?;
        let size = self.effective_font_size(font_size);
        let scale = PxScale::from(size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, font, text);
        let text_h = text_h + text_h / 3;
        Ok((text_w, text_h))
    }

    /// Calculate text dimensions using default font size
    pub fn text_size(&self, text: &str) -> Result<(u32, u32)> {
        self.text_size_with(text, None)
    }

    /// Get total box size including padding with optional font size override
    pub fn box_size_with(
        &self,
        text: &str,
        mode: &TextStyleMode,
        font_size: Option<f32>,
    ) -> Result<(u32, u32)> {
        let (text_w, text_h) = self.text_size_with(text, font_size)?;
        let padding = mode.padding() as u32;
        Ok((text_w + padding * 2, text_h + padding * 2))
    }

    /// Get total box size including padding using default font size
    pub fn box_size(&self, text: &str, mode: &TextStyleMode) -> Result<(u32, u32)> {
        self.box_size_with(text, mode, None)
    }

    /// Render text with full style support and optional font size override
    #[allow(clippy::too_many_arguments)]
    pub fn render_styled_with(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: f32,
        y: f32,
        text_color: Color,
        bg_fill_color: Color,
        bg_outline_color: Color,
        mode: TextStyleMode,
        draw_fill: bool,
        draw_outline: bool,
        thickness: usize,
        font_size: Option<f32>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let font = self.get_font()?;
        let size = self.effective_font_size(font_size);
        let scale = PxScale::from(size);
        let (text_w, text_h) = imageproc::drawing::text_size(scale, font, text);
        let text_h = text_h + text_h / 3;

        // Calculate padding and total box size
        let padding = mode.padding() as u32;
        let box_w = text_w + padding * 2;
        let box_h = text_h + padding * 2;

        let (left, top) = self.calculate_position(x, y, box_w, box_h, img.width());
        let radius = mode.corner_radius();

        // Draw fill if enabled (with alpha blending)
        if draw_fill {
            if mode.is_rounded() {
                self.draw_rounded_rect_filled_blend(
                    img,
                    left,
                    top,
                    box_w,
                    box_h,
                    radius,
                    bg_fill_color,
                );
            } else {
                self.draw_rect_filled_blend(img, left, top, box_w, box_h, bg_fill_color);
            }
        }

        // Draw outline if enabled and thickness > 0
        if draw_outline && thickness > 0 {
            if mode.is_rounded() {
                self.draw_rounded_rect_outline(
                    img,
                    left,
                    top,
                    box_w,
                    box_h,
                    radius,
                    bg_outline_color,
                    thickness,
                );
            } else {
                self.draw_rect_outline(img, left, top, box_w, box_h, bg_outline_color, thickness);
            }
        }

        // Draw text (offset by padding)
        let text_left = left + padding as i32;
        let text_top = top + padding as i32 + self.calculate_text_offset_with(size);

        imageproc::drawing::draw_text_mut(
            img,
            Rgba(text_color.into()),
            text_left,
            text_top,
            scale,
            font,
            text,
        );

        Ok(())
    }

    /// Render text with full style support using default font size
    #[allow(clippy::too_many_arguments)]
    pub fn render_styled(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: f32,
        y: f32,
        text_color: Color,
        bg_fill_color: Color,
        bg_outline_color: Color,
        mode: TextStyleMode,
        draw_fill: bool,
        draw_outline: bool,
        thickness: usize,
    ) -> Result<()> {
        self.render_styled_with(
            img,
            text,
            x,
            y,
            text_color,
            bg_fill_color,
            bg_outline_color,
            mode,
            draw_fill,
            draw_outline,
            thickness,
            None,
        )
    }

    /// Legacy render method for backward compatibility
    pub fn render(
        &self,
        img: &mut RgbaImage,
        text: &str,
        x: f32,
        y: f32,
        color: Color,
        background_color: Color,
    ) -> Result<()> {
        // Default: fill only, no outline
        self.render_styled(
            img,
            text,
            x,
            y,
            color,
            background_color,
            Color::transparent(),
            TextStyleMode::Rect { padding: 0.0 },
            true,  // draw_fill
            false, // draw_outline
            0,
        )
    }

    /// Draw rectangle outline with thickness
    #[allow(clippy::too_many_arguments)]
    fn draw_rect_outline(
        &self,
        img: &mut RgbaImage,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
        color: Color,
        thickness: usize,
    ) {
        let rgba = Rgba(color.into());
        for t in 0..thickness {
            let t = t as i32;
            // Top edge
            for x in (left - t)..=(left + width as i32 + t) {
                if x >= 0 && x < img.width() as i32 && (top - t - 1) >= 0 {
                    img.put_pixel(x as u32, (top - t - 1) as u32, rgba);
                }
            }
            // Bottom edge
            for x in (left - t)..=(left + width as i32 + t) {
                if x >= 0
                    && x < img.width() as i32
                    && (top + height as i32 + t) < img.height() as i32
                {
                    img.put_pixel(x as u32, (top + height as i32 + t) as u32, rgba);
                }
            }
            // Left edge
            for y in (top - t)..=(top + height as i32 + t) {
                if y >= 0 && y < img.height() as i32 && (left - t - 1) >= 0 {
                    img.put_pixel((left - t - 1) as u32, y as u32, rgba);
                }
            }
            // Right edge
            for y in (top - t)..=(top + height as i32 + t) {
                if y >= 0
                    && y < img.height() as i32
                    && (left + width as i32 + t) < img.width() as i32
                {
                    img.put_pixel((left + width as i32 + t) as u32, y as u32, rgba);
                }
            }
        }
    }

    /// Blend a color with alpha onto an existing pixel
    fn blend_pixel(img: &mut RgbaImage, x: u32, y: u32, color: Color) {
        let [cr, cg, cb, ca] = <[u8; 4]>::from(color);
        if ca == 0 {
            return; // Fully transparent, nothing to do
        }
        if ca == 255 {
            img.put_pixel(x, y, Rgba([cr, cg, cb, ca]));
            return;
        }

        let src = img.get_pixel(x, y);
        let [sr, sg, sb, sa] = src.0;

        let alpha = ca as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;

        let nr = (cr as f32 * alpha + sr as f32 * inv_alpha) as u8;
        let ng = (cg as f32 * alpha + sg as f32 * inv_alpha) as u8;
        let nb = (cb as f32 * alpha + sb as f32 * inv_alpha) as u8;
        let na = (ca as f32 + sa as f32 * inv_alpha).min(255.0) as u8;

        img.put_pixel(x, y, Rgba([nr, ng, nb, na]));
    }

    /// Draw filled rectangle with alpha blending
    fn draw_rect_filled_blend(
        &self,
        img: &mut RgbaImage,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
        color: Color,
    ) {
        let x_start = left.max(0) as u32;
        let x_end = ((left + width as i32) as u32).min(img.width());
        let y_start = top.max(0) as u32;
        let y_end = ((top + height as i32) as u32).min(img.height());

        for y in y_start..y_end {
            for x in x_start..x_end {
                Self::blend_pixel(img, x, y, color);
            }
        }
    }

    /// Draw filled rounded rectangle with alpha blending
    #[allow(clippy::too_many_arguments)]
    fn draw_rounded_rect_filled_blend(
        &self,
        img: &mut RgbaImage,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
        radius: f32,
        color: Color,
    ) {
        let r = radius.min(width as f32 / 2.0).min(height as f32 / 2.0) as i32;

        // Fill main rectangle body (excluding corners)
        for y in top..(top + height as i32) {
            for x in left..(left + width as i32) {
                if x < 0 || x >= img.width() as i32 || y < 0 || y >= img.height() as i32 {
                    continue;
                }

                let in_corner = |cx: i32, cy: i32| -> bool {
                    let dx = (x - cx) as f32;
                    let dy = (y - cy) as f32;
                    dx * dx + dy * dy <= (r as f32) * (r as f32)
                };

                // Check if inside any corner circle or main body
                let in_top_left = x < left + r && y < top + r;
                let in_top_right = x >= left + width as i32 - r && y < top + r;
                let in_bottom_left = x < left + r && y >= top + height as i32 - r;
                let in_bottom_right = x >= left + width as i32 - r && y >= top + height as i32 - r;

                let should_draw = if in_top_left {
                    in_corner(left + r, top + r)
                } else if in_top_right {
                    in_corner(left + width as i32 - r - 1, top + r)
                } else if in_bottom_left {
                    in_corner(left + r, top + height as i32 - r - 1)
                } else if in_bottom_right {
                    in_corner(left + width as i32 - r - 1, top + height as i32 - r - 1)
                } else {
                    true
                };

                if should_draw {
                    Self::blend_pixel(img, x as u32, y as u32, color);
                }
            }
        }
    }

    /// Draw rounded rectangle outline with thickness
    #[allow(clippy::too_many_arguments)]
    fn draw_rounded_rect_outline(
        &self,
        img: &mut RgbaImage,
        left: i32,
        top: i32,
        width: u32,
        height: u32,
        radius: f32,
        color: Color,
        thickness: usize,
    ) {
        let rgba = Rgba(color.into());
        let r = radius.min(width as f32 / 2.0).min(height as f32 / 2.0);

        for t in 0..thickness {
            let offset = t as i32;
            let current_r = r + offset as f32;

            // Draw corner arcs
            let corners = [
                (left + r as i32, top + r as i32),                     // top-left
                (left + width as i32 - r as i32 - 1, top + r as i32),  // top-right
                (left + r as i32, top + height as i32 - r as i32 - 1), // bottom-left
                (
                    left + width as i32 - r as i32 - 1,
                    top + height as i32 - r as i32 - 1,
                ), // bottom-right
            ];

            for (i, (cx, cy)) in corners.iter().enumerate() {
                let (start_angle, end_angle) = match i {
                    0 => (std::f32::consts::PI, 1.5 * std::f32::consts::PI), // top-left
                    1 => (1.5 * std::f32::consts::PI, 2.0 * std::f32::consts::PI), // top-right
                    2 => (0.5 * std::f32::consts::PI, std::f32::consts::PI), // bottom-left
                    _ => (0.0, 0.5 * std::f32::consts::PI),                  // bottom-right
                };

                let steps = (current_r * 2.0) as i32;
                for step in 0..=steps {
                    let angle =
                        start_angle + (end_angle - start_angle) * (step as f32 / steps as f32);
                    let px = (*cx as f32 + current_r * angle.cos()).round() as i32;
                    let py = (*cy as f32 + current_r * angle.sin()).round() as i32;
                    if px >= 0 && px < img.width() as i32 && py >= 0 && py < img.height() as i32 {
                        img.put_pixel(px as u32, py as u32, rgba);
                    }
                }
            }

            // Draw straight edges
            // Top edge
            for x in (left + r as i32)..(left + width as i32 - r as i32) {
                let y = top - offset - 1;
                if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                    img.put_pixel(x as u32, y as u32, rgba);
                }
            }
            // Bottom edge
            for x in (left + r as i32)..(left + width as i32 - r as i32) {
                let y = top + height as i32 + offset;
                if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                    img.put_pixel(x as u32, y as u32, rgba);
                }
            }
            // Left edge
            for y in (top + r as i32)..(top + height as i32 - r as i32) {
                let x = left - offset - 1;
                if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                    img.put_pixel(x as u32, y as u32, rgba);
                }
            }
            // Right edge
            for y in (top + r as i32)..(top + height as i32 - r as i32) {
                let x = left + width as i32 + offset;
                if x >= 0 && x < img.width() as i32 && y >= 0 && y < img.height() as i32 {
                    img.put_pixel(x as u32, y as u32, rgba);
                }
            }
        }
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

    fn calculate_text_offset_with(&self, font_size: f32) -> i32 {
        -(font_size / self._scale).floor() as i32 + 1
    }
}
