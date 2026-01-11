# Annotation Style Guide

Comprehensive guide to all annotation styles in `usls` viz module.

## Quick Start

```shell
# Run all annotation demos (default)
cargo run -F vision --example annotator

# Run specific annotation demos
cargo run -F vision --example annotator -- hbb
cargo run -F vision --example annotator -- keypoint
cargo run -F vision --example annotator -- polygon
cargo run -F vision --example annotator -- prob
cargo run -F vision --example annotator -- mask
```

---

## Results

| Type      | Example                                                                 |
|-----------|-------------------------------------------------------------------------|
| HBB       | <img src='https://github.com/jamjamjon/assets/releases/download/images/hbb-styles.png' width='400'> |
| HBB-Text-Loc | <img src='https://github.com/jamjamjon/assets/releases/download/images/hbb-text-loc.png' width='400'> |
| Polygon   | <img src='https://github.com/jamjamjon/assets/releases/download/images/polygon-styles.png' width='400'> |
| Keypoint  | <img src='https://github.com/jamjamjon/assets/releases/download/images/kpt-styles.png' width='400'> |
| Skeleton  | <img src='https://github.com/jamjamjon/assets/releases/download/images/skeleton.png' width='400'> |
| Prob      | <img src='https://github.com/jamjamjon/assets/releases/download/images/prob-styles.png' width='400'> |
| Y         | <img src='https://github.com/jamjamjon/assets/releases/download/images/y.jpg' width='400'> |

---

## Architecture Overview

```
Annotator
├── hbb_style: HbbStyle          // Horizontal bounding box
├── obb_style: ObbStyle          // Oriented bounding box
├── keypoint_style: KeypointStyle // Keypoints & skeleton
├── polygon_style: PolygonStyle   // Polygon/contour
├── mask_style: MaskStyle         // Segmentation mask
├── prob_style: ProbStyle         // Classification probs
└── text_renderer: TextRenderer   // Font & text rendering
```

---

## Drawable Summary

| Drawable | Container | Style | StyleMode | Default TextLoc |
|----------|-----------|-------|-----------|-----------------|
| `Hbb` | `Vec<Hbb>`, `Y` | `HbbStyle` | `HbbStyleMode` | `OuterTopLeft` |
| `Obb` | `Vec<Obb>`, `Y` | `ObbStyle` | `ObbStyleMode` | `OuterTopRight` |
| `Keypoint` | `Vec<Keypoint>`, `Y` | `KeypointStyle` | `KeypointStyleMode` | `OuterTopRight` |
| `Polygon` | `Vec<Polygon>`, `Y` | `PolygonStyle` | - | `Center` |
| `Mask` | `Vec<Mask>`, `Y` | `MaskStyle` | `MaskStyleMode` | - |
| `Prob` | `Vec<Prob>`, `Y` | `ProbStyle` | - | `InnerTopLeft` |

---

## HbbStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide HBB |
| `text_visible` | `bool` | `true` | Show/hide text label |
| `draw_fill` | `bool` | `false` | Fill interior |
| `draw_outline` | `bool` | `true` | Draw border |
| `fill_color` | `ColorSource` | `Auto` | Fill color |
| `outline_color` | `ColorSource` | `Auto` | Border color |
| `mode` | `HbbStyleMode` | `Solid` | Drawing mode |
| `thickness` | `usize` | `3` | Line width (px) |
| `thickness_max_ratio` | `f32` | `0.3` | Max thickness ratio |
| `thickness_direction` | `ThicknessDirection` | `Outward` | Expansion direction |
| `text_style` | `TextStyle` | - | Text configuration |
| `palette` | `Vec<Color>` | base_20 | Color palette |

### HbbStyleMode

| Mode | Factory | Parameters | Description |
|------|---------|------------|-------------|
| `Solid` | `HbbStyle::default()` | - | Solid rectangle |
| `Dashed` | `HbbStyle::dashed()` | `length: 10, gap: 5` | Dashed line |
| `Corners` | `HbbStyle::corners()` | `ratio_long: 0.2, ratio_short: 0.2` | Corner brackets |
| `Rounded` | `HbbStyle::rounded()` | `ratio: 0.1` | Rounded corners |

### ThicknessDirection

| Direction | Description |
|-----------|-------------|
| `Outward` | Expand outward (default) |
| `Inward` | Expand inward |
| `Centered` | Expand both ways |

### Usage

```rust
// Solid (default)
HbbStyle::default()

// Dashed
HbbStyle::dashed()
    .with_mode(HbbStyleMode::Dashed { length: 15, gap: 8 })

// Corners
HbbStyle::corners()
    .with_mode(HbbStyleMode::Corners { ratio_long: 0.25, ratio_short: 0.15 })

// Rounded
HbbStyle::rounded()
    .with_mode(HbbStyleMode::Rounded { ratio: 0.15 })

// Full customization
HbbStyle::default()
    .with_thickness(5)
    .with_draw_fill(true)
    .with_fill_color(ColorSource::Custom(Color::red().with_alpha(80)))
    .with_outline_color(ColorSource::Custom(Color::red()))
    .with_thickness_direction(ThicknessDirection::Inward)
    .with_text_style(TextStyle::default().with_loc(TextLoc::Center))
    .show_confidence(true)
    .show_id(false)
    .show_name(true)
```

---

## ObbStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide OBB |
| `text_visible` | `bool` | `true` | Show/hide text |
| `draw_fill` | `bool` | `false` | Fill interior |
| `draw_outline` | `bool` | `true` | Draw border |
| `fill_color` | `ColorSource` | `Auto` | Fill color |
| `outline_color` | `ColorSource` | `Auto` | Border color |
| `mode` | `ObbStyleMode` | `Rounded` | Drawing mode |
| `thickness` | `usize` | `3` | Line width |
| `text_style` | `TextStyle` | - | Text configuration |

### ObbStyleMode

| Mode | Factory | Parameters | Description |
|------|---------|------------|-------------|
| `Solid` | - | - | Solid lines |
| `Dashed` | `ObbStyle::dashed()` | `length: 10, gap: 5` | Dashed lines |
| `Corners` | - | `ratio_long, ratio_short` | Corner brackets |
| `Rounded` | default | `ratio: 0.1` | Rounded corners |

### Usage

```rust
ObbStyle::default()
    .with_thickness(4)
    .with_mode(ObbStyleMode::Solid)
    .show_confidence(true)
```

---

## KeypointStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide keypoint |
| `text_visible` | `bool` | `true` | Show/hide text |
| `draw_fill` | `bool` | `true` | Fill shape |
| `draw_outline` | `bool` | `true` | Draw outline |
| `fill_color` | `ColorSource` | `Auto` | Fill color |
| `outline_color` | `ColorSource` | `Auto` | Outline color |
| `mode` | `KeypointStyleMode` | `Circle` | Shape mode |
| `radius` | `usize` | `4` | Shape radius (px) |
| `thickness` | `usize` | `2` | Outline thickness (extends outward) |
| `skeleton` | `Option<Skeleton>` | `None` | Skeleton connections |
| `skeleton_thickness` | `usize` | `2` | Skeleton line thickness |
| `text_style` | `TextStyle` | - | Text configuration |
| `palette` | `Vec<Color>` | base_20 | Color palette |

### KeypointStyleMode

| Mode | Factory | Parameters | Description |
|------|---------|------------|-------------|
| `Circle` | default | - | Circle |
| `Star` | `KeypointStyleMode::star()` | `points: 5, inner_ratio: 0.5` | Star |
| `Square` | - | - | Square |
| `Diamond` | - | - | Diamond |
| `Triangle` | - | - | Triangle (up) |
| `Cross` | `KeypointStyleMode::cross()` | `thickness: 2` | Plus sign |
| `X` | `KeypointStyleMode::x()` | `thickness: 2` | X shape |
| `RoundedSquare` | `KeypointStyleMode::rounded_square()` | `corner_ratio: 0.3` | Rounded square |
| `Glow` | `KeypointStyleMode::glow()` | `glow_multiplier: 2.0` | Radial glow |

### Usage

```rust
// With skeleton
KeypointStyle::default()
    .with_skeleton(SKELETON_COCO_19.into())
    .with_radius(6)
    .with_thickness(3)
    .with_skeleton_thickness(2)
    .with_mode(KeypointStyleMode::Circle)
    .show_id(true)
    .show_name(false)

// Star shape with thick outline
KeypointStyle::star()
    .with_mode(KeypointStyleMode::Star { points: 6, inner_ratio: 0.4 })
    .with_radius(10)
    .with_thickness(4)

// Diamond shape
KeypointStyle::default()
    .with_mode(KeypointStyleMode::Diamond)
    .with_radius(8)
    .with_thickness(3)

// Glow effect
KeypointStyle::glow()
    .with_mode(KeypointStyleMode::glow_with(3.0))
```

---

## PolygonStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide |
| `text_visible` | `bool` | `false` | Show/hide text |
| `draw_fill` | `bool` | `true` | Fill polygon |
| `draw_outline` | `bool` | `true` | Draw outline |
| `fill_color` | `ColorSource` | `Custom(white@100)` | Fill color |
| `outline_color` | `ColorSource` | `Custom(black)` | Outline color |
| `thickness` | `usize` | `3` | Line width |
| `background_overlay` | `Option<Color>` | `white@120` | Background overlay |
| `text_style` | `TextStyle` | - | Text (loc=Center) |

### Usage

```rust
PolygonStyle::default()
    .with_text_visible(true)
    .with_fill_color(ColorSource::Auto)
    .with_outline_color(ColorSource::Custom(Color::white()))
    .with_background_overlay(Color::black().with_alpha(100))
    .show_name(true)
    .show_confidence(true)
```

---

## MaskStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide |
| `mode` | `MaskStyleMode` | `Overlay` | Rendering mode |
| `draw_polygons` | `bool` | `false` | Draw contours |
| `draw_polygon_largest` | `bool` | `false` | Draw largest contour |
| `draw_hbbs` | `bool` | `false` | Draw HBBs |
| `draw_obbs` | `bool` | `false` | Draw OBBs |
| `cutout` | `bool` | `true` | Enable cutout |
| `cutout_original` | `bool` | `false` | Cutout from original |
| `cutout_background_color` | `Color` | `green` | Cutout bg color |
| `colormap256` | `Option<ColorMap256>` | `None` | Custom colormap |

### MaskStyleMode

| Mode | Factory | Parameters | Description |
|------|---------|------------|-------------|
| `Overlay` | default | - | Color overlay |
| `Halo` | `MaskStyle::halo()` | `glow_radius, glow_color` | Supervision-style halo |

### GlowRadius

| Type | Description |
|------|-------------|
| `Pixels(usize)` | Fixed pixel radius |
| `Percent(f32)` | % of mask diagonal |

### Usage

```rust
// Overlay (default)
MaskStyle::default()

// Halo effect
MaskStyle::halo()
    .with_mode(MaskStyleMode::halo_with(0.08, Color::magenta().with_alpha(200)))

// With contour extraction
MaskStyle::default()
    .with_draw_polygon_largest(true)
    .with_draw_hbbs(true)
```

---

## ProbStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show/hide |
| `text_x_pos` | `f32` | `0.05` | X position ratio |
| `text_y_pos` | `f32` | `0.05` | Y position ratio |
| `text_style` | `TextStyle` | - | Text configuration |

### Usage

```rust
ProbStyle::default()
    .with_text_x_pos(0.02)
    .with_text_y_pos(0.02)
    .with_text_style(
        TextStyle::default()
            .with_loc(TextLoc::InnerTopRight)
            .with_mode(TextStyleMode::rounded(6.0, 4.0))
    )
```

---

## TextStyle

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visible` | `bool` | `true` | Show text |
| `loc` | `TextLoc` | varies | Position |
| `mode` | `TextStyleMode` | `Rect(4.0)` | Background shape |
| `font_size` | `Option<f32>` | `None` | Font size (None = use global 24.0) |
| `draw_fill` | `bool` | `true` | Fill background |
| `draw_outline` | `bool` | `false` | Draw border |
| `thickness` | `usize` | `0` | Border width |
| `color` | `ColorSource` | `Auto` | Text color |
| `bg_fill_color` | `ColorSource` | `Auto` | Background fill |
| `bg_outline_color` | `ColorSource` | `Auto` | Background border |
| `confidence` | `bool` | `true` | Show confidence |
| `name` | `bool` | `true` | Show name |
| `id` | `bool` | `true` | Show ID |
| `decimal_places` | `usize` | `3` | Confidence decimals |

### TextLoc (17 positions)

```
┌─────────────────────────────────────────────────────────┐
│ OuterTopLeft     OuterTopCenter     OuterTopRight       │
├─────────────────────────────────────────────────────────┤
│ InnerTopLeft     InnerTopCenter     InnerTopRight       │
│                                                         │
│ InnerCenterLeft  Center             InnerCenterRight    │
│ OuterCenterLeft                     OuterCenterRight    │
│                                                         │
│ InnerBottomLeft  InnerBottomCenter  InnerBottomRight    │
├─────────────────────────────────────────────────────────┤
│ OuterBottomLeft  OuterBottomCenter  OuterBottomRight    │
└─────────────────────────────────────────────────────────┘
```

### TextStyleMode

| Mode | Factory | Parameters | Description |
|------|---------|------------|-------------|
| `Rect` | `TextStyleMode::rect(padding)` | `padding: f32` | Rectangle bg |
| `Rounded` | `TextStyleMode::rounded(padding, radius)` | `padding, radius` | Rounded bg |

### Usage

```rust
TextStyle::default()
    .with_loc(TextLoc::InnerTopLeft)
    .with_mode(TextStyleMode::rounded(5.0, 3.0))
    .with_font_size(28.0)  // Override global font size
    .with_draw_fill(true)
    .with_draw_outline(true)
    .with_thickness(2)
    .with_color(ColorSource::Custom(Color::white()))
    .with_bg_fill_color(ColorSource::InheritOutlineAlpha(200))
    .with_bg_outline_color(ColorSource::Custom(Color::black()))
    .with_confidence(true)
    .with_name(true)
    .with_id(false)
    .with_decimal_places(2)
```

---

## ColorSource

| Source | Description |
|--------|-------------|
| `Auto` | Auto from palette (shapes) or black (text) |
| `AutoAlpha(u8)` | Auto with custom alpha |
| `InheritOutline` | Inherit shape's outline color |
| `InheritOutlineAlpha(u8)` | Inherit outline + custom alpha |
| `InheritFill` | Inherit shape's fill color |
| `InheritFillAlpha(u8)` | Inherit fill + custom alpha |
| `Custom(Color)` | Specific color |

### Usage

```rust
// Text background inherits outline color with alpha
TextStyle::default()
    .with_bg_fill_color(ColorSource::InheritOutlineAlpha(180))

// Custom colors
HbbStyle::default()
    .with_outline_color(ColorSource::Custom(Color::red()))
    .with_fill_color(ColorSource::Custom(Color::red().with_alpha(60)))
```

---

## Color

### Predefined Colors

```rust
Color::black()      // RGB(0, 0, 0)
Color::white()      // RGB(255, 255, 255)
Color::red()        // RGB(255, 0, 0)
Color::green()      // RGB(0, 255, 0)
Color::blue()       // RGB(0, 0, 255)
Color::yellow()     // RGB(255, 255, 0)
Color::cyan()       // RGB(0, 255, 255)
Color::magenta()    // RGB(255, 0, 255)
Color::transparent() // RGBA(0, 0, 0, 0)
```

### Palettes

```rust
Color::palette_base_20()        // 20 distinct colors
Color::palette_rainbow_10()     // Rainbow palette
Color::palette_coco_80()        // COCO dataset colors
Color::palette_pascal_voc_21()  // Pascal VOC colors
Color::palette_ade20k_150()     // ADE20K colors
Color::palette_rand(n)          // Random n colors
```

### Custom Colors

```rust
// From RGBA
Color::from_rgba(255, 128, 0, 255)
Color::from([255, 128, 0, 255])
Color::from((255, 128, 0, 255))

// From hex string
"#ff8800".parse::<Color>()?
"#ff8800ff".parse::<Color>()?

// Modify alpha
Color::red().with_alpha(128)
```

---

## Per-Instance Style Override

Each drawable can have its own style that overrides the annotator's default:

```rust
// HBB with custom style
Hbb::default()
    .with_xyxy(100.0, 100.0, 300.0, 200.0)
    .with_name("custom")
    .with_style(
        HbbStyle::corners()
            .with_thickness(6)
            .with_outline_color(ColorSource::Custom(Color::magenta()))
    )

// Keypoint with custom style
Keypoint::default()
    .with_xy(150.0, 150.0)
    .with_style(
        KeypointStyle::star()
            .with_radius(15)
    )
```

---

## Annotator Setup

```rust
let annotator = Annotator::default()
    .with_font("path/to/font.ttf")?
    .with_font_size(16.0)
    .with_hbb_style(HbbStyle::default())
    .with_obb_style(ObbStyle::default())
    .with_keypoint_style(KeypointStyle::default())
    .with_polygon_style(PolygonStyle::default())
    .with_mask_style(MaskStyle::default())
    .with_prob_style(ProbStyle::default());

// Annotate and save
annotator.annotate(&image, &drawable)?.save("output.jpg")?;
```

---

## Output Examples

Run the example to generate all demo images:

```shell
cargo run -F vision --example annotate
```

Output files in `runs/Annotate/`:

| File | Description |
|------|-------------|
| `Hbb/styles.png` | HBB modes (Solid, Dashed, Corners, Rounded) + ThicknessDirection (Outward, Inward, Centered) |
| `Hbb/text_loc.png` | All 17 TextLoc positions with boundary cases (edge boxes, tiny box) |
| `Keypoint/styles.png` | All 11 keypoint shapes with smooth anti-aliased outlines (Circle, Star, Square, Diamond, Triangle, Cross, X, RoundedSquare, Glow, Star6, Star8) |
| `Keypoint/skeleton.png` | Pose skeleton with COCO-19 connections and configurable thickness |
| `Polygon/styles.png` | 12 diverse polygon shapes (Star, Arrow, Hexagon, Pentagon, Lightning, House, Triangle, Cross, etc.) with various colors and styles |
| `Prob/styles.png` | Classification positions (TopLeft, TopRight, BottomLeft, BottomRight, Center, Rounded) |
| `Mask/styles.png` | Mask rendering modes (Overlay, Halo with glow effects) |
| `Y/combined.jpg` | Combined Y annotations on real image |
