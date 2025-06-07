use crate::{Color, Style, StyleColors, TextRenderer};

/// Drawing context containing styles and renderers for visualization.
#[derive(Debug, Clone)]
pub struct DrawContext<'a> {
    pub text_renderer: &'a TextRenderer,
    pub prob_style: Option<&'a Style>,
    pub hbb_style: Option<&'a Style>,
    pub obb_style: Option<&'a Style>,
    pub keypoint_style: Option<&'a Style>,
    pub polygon_style: Option<&'a Style>,
    pub mask_style: Option<&'a Style>,
}

impl<'a> DrawContext<'a> {
    pub fn update_style(
        &self,
        instance_style: Option<&'a Style>,
        global_style: Option<&'a Style>,
        id: Option<usize>,
    ) -> Style {
        let style = instance_style.or(global_style).cloned().unwrap_or_default();
        let is_prob = global_style == self.prob_style;
        let is_hbb = global_style == self.hbb_style;
        let is_obb = global_style == self.obb_style;
        let is_keypoint = global_style == self.keypoint_style;
        let is_polygon = global_style == self.polygon_style;
        let is_mask = global_style == self.mask_style;

        let mut color = StyleColors::default();

        if is_hbb || is_obb {
            color = self.compute_hbb_obb_colors(&style, id);
        } else if is_keypoint {
            color = self.compute_keypoint_colors(&style, id);
        } else if is_polygon || is_mask {
            color = self.compute_polygon_colors(&style, id);
        } else if is_prob {
            color = self.compute_prob_colors(&style, id);
        }

        style.with_color(color)
    }

    fn compute_prob_colors(&self, style: &Style, id: Option<usize>) -> StyleColors {
        let color_text = style.color().text().copied().unwrap_or_else(Color::black);
        let color_text_bg = style.color().text_bg().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id))
                .unwrap_or_else(|| Color::white().with_alpha(70))
        });

        StyleColors::default()
            .with_text(color_text)
            .with_text_bg(color_text_bg)
    }

    fn compute_hbb_obb_colors(&self, style: &Style, id: Option<usize>) -> StyleColors {
        // Get fill color following these rules:
        // 1. Use style's fill color if set.
        // 2. Default to white with 100 alpha.
        let mut color_fill = style
            .color()
            .fill()
            .copied()
            .unwrap_or_else(|| Color::white().with_alpha(80));

        // set custom alpha
        if let Some(a) = style.color_fill_alpha() {
            color_fill = color_fill.with_alpha(a);
        }

        // Get outline color by:
        // 1. Using style's outline color if set.
        // 2. Using palette color via instance ID if available.
        // 3. Defaulting to black.
        let color_outline = style.color().outline().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id))
                .unwrap_or_else(Color::black)
        });

        // Retrieves the text color from the provided style. If the text color is not set,
        // it defaults to black. The color is copied to ensure ownership.
        let color_text = style.color().text().copied().unwrap_or_else(Color::black);

        // Determines the background color for text based on the provided style and optional ID.
        // - If the style specifies a background color, it is used directly.
        // - If no background color is specified in the style:
        //   - If an ID is provided, the color is derived from the palette using the ID.
        //   - Otherwise, a default white color is used.
        let color_text_bg = style.color().text_bg().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id))
                .unwrap_or_else(Color::white)
        });

        StyleColors::default()
            .with_fill(color_fill)
            .with_outline(color_outline)
            .with_text(color_text)
            .with_text_bg(color_text_bg)
    }

    fn compute_keypoint_colors(&self, style: &Style, id: Option<usize>) -> StyleColors {
        // Color handling strategy:
        // 1. If style color is set, use the style color
        // 2. Otherwise, if instance has ID, use ID with palette
        // 3. Otherwise, use fixed black color
        // 4. And set color alpha
        let mut color_fill = style.color().fill().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id).with_alpha(220))
                .unwrap_or_else(|| Color::black().with_alpha(220))
        });

        // set custom alpha
        if let Some(a) = style.color_fill_alpha() {
            color_fill = color_fill.with_alpha(a);
        }

        // Color handling strategy:
        // 1. If style color is set, use the style color
        // 2. Otherwise, use fixed white color
        let color_outline = style
            .color()
            .outline()
            .copied()
            .unwrap_or_else(Color::white);

        // Text color strategy:
        // 1. Use user-defined color if set, otherwise use black
        let color_text = style.color().text().copied().unwrap_or_else(Color::black);

        // Text background color strategy:
        // 1. Use user-defined color if set
        // 2. Use palette color based on ID
        // 3. Default to white background
        let color_text_bg = style.color().text_bg().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id))
                .unwrap_or_else(Color::white)
        });

        StyleColors::default()
            .with_fill(color_fill)
            .with_outline(color_outline)
            .with_text(color_text)
            .with_text_bg(color_text_bg)
    }

    fn compute_polygon_colors(&self, style: &Style, id: Option<usize>) -> StyleColors {
        // Color handling strategy:
        // 1. If style color is set, use the style color
        // 2. Otherwise, if instance has ID, use ID with palette
        // 3. Otherwise, use fixed black color
        // 4. And set color alpha
        let mut color_fill = style.color().fill().copied().unwrap_or_else(|| {
            id.map(|id| style.color_from_palette(id).with_alpha(79))
                .unwrap_or_else(|| Color::black().with_alpha(79))
        });

        // set custom alpha
        if let Some(a) = style.color_fill_alpha() {
            color_fill = color_fill.with_alpha(a);
        }
        // Color handling strategy:
        // 1. If style color is set, use the style color
        // 2. Otherwise, use fixed white color
        let color_outline = style
            .color()
            .outline()
            .copied()
            .unwrap_or_else(Color::white);

        // Text color strategy:
        // 1. Use user-defined color if set, otherwise use black
        let color_text = style.color().text().copied().unwrap_or_else(Color::black);

        // Text background color strategy:
        // 1. Use user-defined color if set
        // 2. Use palette color based on ID
        // 3. Default to white background
        let color_text_bg = style
            .color()
            .text_bg()
            .copied()
            .unwrap_or_else(Color::white);

        StyleColors::default()
            .with_fill(color_fill)
            .with_outline(color_outline)
            .with_text(color_text)
            .with_text_bg(color_text_bg)
    }
}
