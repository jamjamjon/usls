# Annotator

Visualizes model results with customizable styles.

!!! example "Quick Start"
    ```rust
    use usls::*;

    let annotator = Annotator::default();
    let ys = model.run(&images)?;

    for (img, y) in images.iter().zip(ys.iter()) {
        annotator.annotate(img, y)?.save("output.jpg")?;
    }
    ```

---

## Supported Types

| Type | Style | Key Methods |
| :--- | :--- | :--- |
| HBB | `HbbStyle` | `thickness`, `draw_fill`, `draw_outline`, `mode` |
| OBB | `ObbStyle` | `thickness`, `draw_fill`, `mode` |
| Keypoint | `KeypointStyle` | `radius`, `skeleton`, `skeleton_thickness`, `mode` |
| Polygon | `PolygonStyle` | `thickness`, `draw_fill`, `background_overlay` |
| Mask | `MaskStyle` | `mode` (`Overlay`/`Halo`), `draw_polygons` |
| Prob | `ProbStyle` | `text_x_pos`, `text_y_pos` |

---

## HbbStyle

**Modes**: `Solid` (default), `Dashed`, `Corners`, `Rounded`

!!! example "Example"
    ```rust
    use usls::viz::*;

    // Dashed box
    let style = HbbStyle::dashed()
        .with_thickness(3)
        .with_mode(HbbStyleMode::Dashed { length: 15, gap: 8 });

    // Corners only
    let style = HbbStyle::corners()
        .with_mode(HbbStyleMode::Corners { ratio_long: 0.25, ratio_short: 0.15 });
    ```

---

## KeypointStyle

**Modes**: `Circle` (default), `Star`, `Square`, `Diamond`, `Triangle`, `Cross`, `X`, `RoundedSquare`, `Glow`

!!! example "Example"
    ```rust
    use usls::viz::*;

    // With skeleton
    let style = KeypointStyle::default()
        .with_skeleton(SKELETON_COCO_19.into())
        .with_radius(6)
        .with_skeleton_thickness(2);

    // Star shape
    let style = KeypointStyle::star()
        .with_mode(KeypointStyleMode::Star { points: 6, inner_ratio: 0.4 });
    ```

---

## MaskStyle

**Modes**: `Overlay` (default), `Halo`

!!! example "Example"
    ```rust
    use usls::viz::*;

    // Halo effect
    let style = MaskStyle::halo()
        .with_mode(MaskStyleMode::halo_with(0.08, Color::magenta().with_alpha(200)));

    // With contours
    let style = MaskStyle::default()
        .with_draw_polygon_largest(true)
        .with_draw_hbbs(true);
    ```

---

## TextStyle

**Locations**: `OuterTopLeft`, `OuterTopRight`, `InnerTopLeft`, `InnerTopRight`, `Center`, etc.

**Modes**: `Rect(padding)`, `Rounded(padding, radius)`

!!! example "Example"
    ```rust
    use usls::viz::*;

    let style = TextStyle::default()
        .with_loc(TextLoc::InnerTopLeft)
        .with_mode(TextStyleMode::rounded(5.0, 3.0))
        .with_confidence(true)
        .with_name(true)
        .with_id(false);
    ```

---

## ColorSource

| Source | Description |
| :--- | :--- |
| `Auto` | Auto from palette |
| `AutoAlpha(u8)` | Auto with custom alpha |
| `InheritOutline` | Inherit shape outline color |
| `InheritFill` | Inherit shape fill color |
| `Custom(Color)` | Specific color |

!!! example "Example"
    ```rust
    use usls::viz::*;

    HbbStyle::default()
        .with_outline_color(ColorSource::Custom(Color::red()))
        .with_fill_color(ColorSource::Custom(Color::red().with_alpha(60)));
    ```

---

## Complete Example

!!! example "Custom Annotator"
    ```rust
    use usls::*;
    use usls::viz::*;

    fn main() -> anyhow::Result<()> {
        let config = Config::default().commit()?;
        let mut model = YOLO::new(config)?;

        let annotator = Annotator::default()
            .with_hbb_style(
                HbbStyle::default()
                    .with_thickness(2)
                    .with_outline_color(Color::Green)
            )
            .with_text_style(
                TextStyle::default()
                    .with_color(Color::White)
                    .with_bg_fill_color(Color::Black.with_alpha(0.5))
            );

        let dl = DataLoader::new("image.jpg")?.stream()?;
        
        for (images, _) in dl {
            let ys = model.run(&images)?;
            for (img, y) in images.iter().zip(ys.iter()) {
                annotator.annotate(img, y)?.save("output.jpg")?;
            }
        }

        Ok(())
    }
    ```

---

## Per-Instance Override

Each result can have its own style:

!!! example "Example"
    ```rust
    use usls::viz::*;

    Hbb::default()
        .with_xyxy(100.0, 100.0, 300.0, 200.0)
        .with_name("custom")
        .with_style(
            HbbStyle::corners()
                .with_thickness(6)
                .with_outline_color(ColorSource::Custom(Color::magenta()))
        );
    ```
