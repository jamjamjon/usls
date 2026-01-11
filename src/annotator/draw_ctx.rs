use crate::{
    Color, ColorSource, HbbStyle, KeypointStyle, MaskStyle, ObbStyle, Palette, PolygonStyle,
    ProbStyle, TextRenderer, TextStyle,
};

/// Drawing context containing styles and renderers for visualization.
#[derive(Debug, Clone)]
pub struct DrawContext<'a> {
    pub text_renderer: &'a TextRenderer,
    pub prob_style: Option<&'a ProbStyle>,
    pub hbb_style: Option<&'a HbbStyle>,
    pub obb_style: Option<&'a ObbStyle>,
    pub keypoint_style: Option<&'a KeypointStyle>,
    pub polygon_style: Option<&'a PolygonStyle>,
    pub mask_style: Option<&'a MaskStyle>,
}

/// Resolved colors for drawing (computed from Style + palette + ID)
#[derive(Debug, Clone, Default)]
pub struct ResolvedColors {
    pub fill: Color,
    pub outline: Color,
    pub text: Color,
    pub text_bg_fill: Color,
    pub text_bg_outline: Color,
}

impl DrawContext<'_> {
    /// Resolve colors for HBB drawing
    pub fn resolve_hbb_colors(&self, style: &HbbStyle, id: Option<usize>) -> ResolvedColors {
        let palette_color = id.map(|i| style.color_from_palette(i));

        let outline = self.resolve_color(*style.outline_color(), palette_color, Color::black());
        let fill = self.resolve_color(
            *style.fill_color(),
            Some(Color::white().with_alpha(80)),
            Color::white().with_alpha(80),
        );

        self.resolve_text_colors(style.text_style(), outline, fill, palette_color)
    }

    /// Resolve colors for OBB drawing
    pub fn resolve_obb_colors(&self, style: &ObbStyle, id: Option<usize>) -> ResolvedColors {
        let palette_color = id.map(|i| style.color_from_palette(i));

        let outline = self.resolve_color(*style.outline_color(), palette_color, Color::black());
        let fill = self.resolve_color(
            *style.fill_color(),
            Some(Color::white().with_alpha(80)),
            Color::white().with_alpha(80),
        );

        self.resolve_text_colors(style.text_style(), outline, fill, palette_color)
    }

    /// Resolve colors for keypoint drawing
    pub fn resolve_keypoint_colors(
        &self,
        style: &KeypointStyle,
        id: Option<usize>,
    ) -> ResolvedColors {
        let palette_color = id.map(|i| style.color_from_palette(i).with_alpha(220));

        let fill = self.resolve_color(
            *style.fill_color(),
            palette_color,
            Color::black().with_alpha(220),
        );
        let outline =
            self.resolve_color(*style.outline_color(), Some(Color::white()), Color::white());

        self.resolve_text_colors(
            style.text_style(),
            outline,
            fill,
            id.map(|i| style.color_from_palette(i)),
        )
    }

    /// Resolve colors for polygon drawing
    pub fn resolve_polygon_colors(
        &self,
        style: &PolygonStyle,
        id: Option<usize>,
    ) -> ResolvedColors {
        let palette_color = id.map(|i| style.color_from_palette(i).with_alpha(79));

        let fill = self.resolve_color(
            *style.fill_color(),
            palette_color,
            Color::black().with_alpha(79),
        );
        let outline =
            self.resolve_color(*style.outline_color(), Some(Color::white()), Color::white());

        self.resolve_text_colors(
            style.text_style(),
            outline,
            fill,
            id.map(|i| style.color_from_palette(i)),
        )
    }

    /// Resolve colors for prob drawing (text only)
    pub fn resolve_prob_colors(&self, style: &ProbStyle, id: Option<usize>) -> ResolvedColors {
        let palette_color = id.map(|i| style.color_from_palette(i));
        let text_style = style.text_style();

        // For Prob, use palette_color for both fill and outline inheritance
        // since there's no actual shape to inherit from
        let inherit_color = palette_color.unwrap_or_else(|| Color::white().with_alpha(70));

        let text = self.resolve_text_color_source(
            *text_style.color(),
            inherit_color,  // outline -> palette color
            inherit_color,  // fill -> palette color
            Color::black(), // default for text is black
        );
        let text_bg_fill = self.resolve_text_color_source(
            text_style.bg_color(),
            inherit_color,
            inherit_color,
            inherit_color,
        );
        let text_bg_outline = self.resolve_text_color_source(
            *text_style.bg_outline_color(),
            inherit_color,
            inherit_color,
            Color::transparent(),
        );

        ResolvedColors {
            fill: Color::transparent(),
            outline: Color::transparent(),
            text,
            text_bg_fill,
            text_bg_outline,
        }
    }

    /// Resolve a ColorSource to actual Color
    fn resolve_color(
        &self,
        source: ColorSource,
        palette_color: Option<Color>,
        default: Color,
    ) -> Color {
        match source {
            ColorSource::Auto => palette_color.unwrap_or(default),
            ColorSource::AutoAlpha(alpha) => palette_color.unwrap_or(default).with_alpha(alpha),
            ColorSource::Custom(c) => c,
            // For shape colors, inherit doesn't make sense - use default
            ColorSource::InheritOutline | ColorSource::InheritFill => default,
            ColorSource::InheritOutlineAlpha(alpha) | ColorSource::InheritFillAlpha(alpha) => {
                default.with_alpha(alpha)
            }
        }
    }

    /// Resolve text colors with inheritance support
    fn resolve_text_colors(
        &self,
        text_style: &TextStyle,
        outline: Color,
        fill: Color,
        palette_color: Option<Color>,
    ) -> ResolvedColors {
        let text =
            self.resolve_text_color_source(*text_style.color(), outline, fill, Color::black());
        let text_bg_fill = self.resolve_text_color_source(
            text_style.bg_color(),
            outline,
            fill,
            palette_color.unwrap_or_else(Color::white),
        );
        let text_bg_outline = self.resolve_text_color_source(
            *text_style.bg_outline_color(),
            outline,
            fill,
            Color::transparent(),
        );

        ResolvedColors {
            fill,
            outline,
            text,
            text_bg_fill,
            text_bg_outline,
        }
    }

    /// Resolve a ColorSource for text (with inheritance)
    fn resolve_text_color_source(
        &self,
        source: ColorSource,
        outline: Color,
        fill: Color,
        default: Color,
    ) -> Color {
        match source {
            ColorSource::Auto => default,
            ColorSource::AutoAlpha(alpha) => default.with_alpha(alpha),
            ColorSource::InheritOutline => outline,
            ColorSource::InheritOutlineAlpha(alpha) => outline.with_alpha(alpha),
            ColorSource::InheritFill => fill,
            ColorSource::InheritFillAlpha(alpha) => fill.with_alpha(alpha),
            ColorSource::Custom(c) => c,
        }
    }
}
