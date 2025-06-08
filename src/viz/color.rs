use anyhow::Result;
use rand::Rng;

/// Color: 0xRRGGBBAA
#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct Color(pub u32);

impl From<u32> for Color {
    fn from(x: u32) -> Self {
        Self(x)
    }
}

impl From<(u8, u8, u8)> for Color {
    fn from((r, g, b): (u8, u8, u8)) -> Self {
        Self::from_rgba(r, g, b, 0xff)
    }
}

impl From<[u8; 3]> for Color {
    fn from(c: [u8; 3]) -> Self {
        Self::from((c[0], c[1], c[2]))
    }
}

impl From<(u8, u8, u8, u8)> for Color {
    fn from((r, g, b, a): (u8, u8, u8, u8)) -> Self {
        Self::from_rgba(r, g, b, a)
    }
}

impl From<[u8; 4]> for Color {
    fn from(c: [u8; 4]) -> Self {
        Self::from((c[0], c[1], c[2], c[3]))
    }
}

impl From<Color> for (u8, u8, u8, u8) {
    fn from(color: Color) -> Self {
        color.rgba()
    }
}

impl From<Color> for [u8; 4] {
    fn from(color: Color) -> Self {
        let (r, g, b, a) = color.rgba();
        [r, g, b, a]
    }
}

impl From<Color> for (u8, u8, u8) {
    fn from(color: Color) -> Self {
        color.rgb()
    }
}

impl From<Color> for [u8; 3] {
    fn from(color: Color) -> Self {
        let (r, g, b) = color.rgb();
        [r, g, b]
    }
}

impl std::str::FromStr for Color {
    type Err = anyhow::Error;

    fn from_str(x: &str) -> Result<Self, Self::Err> {
        let hex = x.trim_start_matches('#');
        let hex = match hex.len() {
            6 => format!("{}ff", hex),
            8 => hex.to_string(),
            _ => {
                return Err(anyhow::anyhow!(
                    "Failed to convert `Color` from str: invalid length"
                ))
            }
        };

        u32::from_str_radix(&hex, 16)
            .map(Self)
            .map_err(|_| anyhow::anyhow!("Failed to convert `Color` from str: invalid hex"))
    }
}

impl Color {
    /// Creates a new Color from RGBA components.
    /// Each component is an 8-bit value (0-255).
    const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32))
    }

    /// Returns the color components as RGBA tuple.
    /// Each component is an 8-bit value (0-255).
    pub fn rgba(&self) -> (u8, u8, u8, u8) {
        let r = ((self.0 >> 24) & 0xff) as u8;
        let g = ((self.0 >> 16) & 0xff) as u8;
        let b = ((self.0 >> 8) & 0xff) as u8;
        let a = (self.0 & 0xff) as u8;
        (r, g, b, a)
    }

    /// Returns the RGB components as a tuple, excluding alpha.
    /// Each component is an 8-bit value (0-255).
    pub fn rgb(&self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.rgba();
        (r, g, b)
    }

    /// Returns the BGR components as a tuple.
    /// Useful for OpenCV-style color formats.
    pub fn bgr(&self) -> (u8, u8, u8) {
        let (r, g, b) = self.rgb();
        (b, g, r)
    }

    /// Returns the red component (0-255).
    pub fn r(&self) -> u8 {
        self.rgba().0
    }

    /// Returns the green component (0-255).
    pub fn g(&self) -> u8 {
        self.rgba().1
    }

    /// Returns the blue component (0-255).
    pub fn b(&self) -> u8 {
        self.rgba().2
    }

    /// Returns the alpha component (0-255).
    pub fn a(&self) -> u8 {
        self.rgba().3
    }

    /// Returns the color as a hex string in the format "#RRGGBBAA".
    pub fn hex(&self) -> String {
        format!("#{:08x}", self.0)
    }

    /// Creates a new color with the specified alpha value while keeping RGB components.
    pub fn with_alpha(self, a: u8) -> Self {
        let (r, g, b) = self.rgb();

        (r, g, b, a).into()
    }

    /// Creates a black color (RGB: 0,0,0) with full opacity.
    pub fn black() -> Color {
        [0, 0, 0, 255].into()
    }

    /// Creates a white color (RGB: 255,255,255) with full opacity.
    pub fn white() -> Color {
        [255, 255, 255, 255].into()
    }

    /// Creates a green color (RGB: 0,255,0) with full opacity.
    pub fn green() -> Color {
        [0, 255, 0, 255].into()
    }

    /// Creates a red color (RGB: 255,0,0) with full opacity.
    pub fn red() -> Color {
        [255, 0, 0, 255].into()
    }

    /// Creates a blue color (RGB: 0,0,255) with full opacity.
    pub fn blue() -> Color {
        [0, 0, 255, 255].into()
    }

    /// Creates a color palette from a slice of convertible values.
    pub fn create_palette<A: Into<Self> + Copy>(xs: &[A]) -> Vec<Self> {
        xs.iter().copied().map(Into::into).collect()
    }

    /// Attempts to create a color palette from hex color strings.
    /// Returns an error if any string is not a valid hex color.
    pub fn try_create_palette(xs: &[&str]) -> Result<Vec<Self>> {
        xs.iter().map(|x| x.parse()).collect()
    }

    /// Creates a palette of random colors with the specified size.
    pub fn palette_rand(n: usize) -> Vec<Self> {
        let mut rng = rand::rng();
        let xs: Vec<(u8, u8, u8)> = (0..n)
            .map(|_| {
                (
                    rng.random_range(0..=255),
                    rng.random_range(0..=255),
                    rng.random_range(0..=255),
                )
            })
            .collect();

        Self::create_palette(&xs)
    }

    /// Returns a predefined palette of 20 base colors.
    pub fn palette_base_20() -> Vec<Self> {
        Self::create_palette(&PALETTE_BASE)
    }

    /// Returns a cotton candy themed palette of 5 colors.
    pub fn palette_cotton_candy_5() -> Result<Vec<Self>> {
        Self::try_create_palette(&["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"])
    }

    /// Returns a tropical sunrise themed palette of 5 colors.
    #[inline(always)]
    pub fn palette_tropical_sunrise_5() -> Result<Vec<Self>> {
        // https://colorkit.co/palette/e12729-f37324-f8cc1b-72b043-007f4e/
        Self::try_create_palette(&["#e12729", "#f37324", "#f8cc1b", "#72b043", "#007f4e"])
    }

    /// Returns a rainbow themed palette of 10 colors.
    pub fn palette_rainbow_10() -> Vec<Self> {
        Self::create_palette(&[
            0xff595eff, 0xff924cff, 0xffca3aff, 0xc5ca30ff, 0x8ac926ff, 0x52a675ff, 0x1982c4ff,
            0x4267acff, 0x6a4c93ff, 0xb5a6c9ff,
        ])
    }

    /// Returns the COCO dataset color palette with 80 colors.
    pub fn palette_coco_80() -> Vec<Self> {
        Self::create_palette(&PALETTE_COCO_80)
    }

    /// Returns the Pascal VOC dataset color palette with 21 colors.
    pub fn palette_pascal_voc_21() -> Vec<Self> {
        Self::create_palette(&PALETTE_PASCAL_VOC_20)
    }
    /// Returns the ADE20K dataset color palette with 150 colors.
    pub fn palette_ade20k_150() -> Vec<Self> {
        Self::create_palette(&PALETTE_ADE20K_150)
    }
}

const PALETTE_BASE: [u32; 38] = [
    0x00ff7fff, // SpringGreen
    0xff69b4ff, // HotPink
    0xff6347ff, // Tomato
    0xffd700ff, // Gold
    0xbc8f8fff, // RosyBrown
    0x00bfffff, // DeepSkyBlue
    0x8fb88fff, // DarkSeaGreen
    0xee82eeff, // Violet
    0x9acd32ff, // YellowGreen
    0xcd853fff, // Peru
    0x1e90ffff, // DodgerBlue
    0xd74a49ff, // ?
    0x7fffd4ff, // AquaMarine
    0x808000ff, // Olive
    0x00ffffff, // Cyan
    0x8a2befff, // BlueViolet
    0xa52a2aff, // Brown
    0xd8bfd8ff, // Thistle
    0xf0ffffff, // Azure
    0x609ea0ff, // CadetBlue
    0xffa500ff, // Orange
    0x800080ff, // Purple
    0x40e0d0ff, // Turquoise
    0x00008bff, // DarkBlue
    0x006400ff, // DarkGreen
    0xd3d3d3ff, // LightGray
    0x2f4f4fff, // DarkSlateGray
    0x8b4513ff, // SaddleBrown
    0xffb6c1ff, // LightPink
    0xffffffff, // White
    0xff4500ff, // OrangeRed
    0x4682b4ff, // SteelBlue
    0x00fa9aff, // MediumSpringGreen
    0xadff2fff, // GreenYellow
    0xff1493ff, // DeepPink
    0xdeb887ff, // BurlyWood
    0xadd8e6ff, // LightBlue
    0x708090ff, // SlateGray
];

const PALETTE_COCO_80: [u32; 80] = [
    0xFF3838FF, 0xFF9D97FF, 0xFF701FFF, 0xFFB21DFF, 0xCFD231FF, 0x48F90AFF, 0x92CC17FF, 0x3DDB86FF,
    0x1A9334FF, 0x00D4BBFF, 0x2C99A8FF, 0x00C2FFFF, 0x344593FF, 0x6473FFFF, 0x0018ECFF, 0x8438FFFF,
    0x520085FF, 0xCB38FFFF, 0xFF95C8FF, 0xFF37C7FF, 0xD7D7D7FF, 0xC8FF00FF, 0xC8D200FF, 0x5CFF00FF,
    0x00FF0DFF, 0x00FF94FF, 0x00FFD5FF, 0x00D5FFFF, 0x0095FFFF, 0x0045FFFF, 0x3000FFFF, 0x8400FFFF,
    0xCB00FFFF, 0xFF00ECFF, 0xFF0094FF, 0xFF0064FF, 0xFF3C00FF, 0xFF6400FF, 0xFFA000FF, 0xFFE700FF,
    0xD2FF00FF, 0xA0FF00FF, 0x64FF00FF, 0x00FF1EFF, 0x00FF64FF, 0x00FFA0FF, 0x00FFE7FF, 0x00A0FFFF,
    0x0064FFFF, 0x001EFFFF, 0x6400FFFF, 0xA000FFFF, 0xFF00D2FF, 0xFF0064FF, 0xFF3000FF, 0xFF5C00FF,
    0xFF9400FF, 0xFFD200FF, 0xD2FF00FF, 0xA0FF00FF, 0x64FF00FF, 0x1EFF00FF, 0x00FF64FF, 0x00FFA0FF,
    0x00FFD2FF, 0x00A0FFFF, 0x0064FFFF, 0x001EFFFF, 0x5A00FFFF, 0x9400FFFF, 0xFF00D2FF, 0xFF0094FF,
    0xFF0064FF, 0xFF3000FF, 0xFF5C00FF, 0xFF9400FF, 0xFFD200FF, 0xD2FF00FF, 0xA0FF00FF, 0x64FF00FF,
];

const PALETTE_PASCAL_VOC_20: [u32; 21] = [
    0x000000FF, 0xFF0000FF, 0x00FF00FF, 0x0000FFFF, 0xFFFF00FF, 0x00FFFFFF, 0xFF00FFFF, 0x800000FF,
    0x008000FF, 0x000080FF, 0x808000FF, 0x800080FF, 0x008080FF, 0xC0C0C0FF, 0xFFA500FF, 0xA52A2AFF,
    0x7FFF00FF, 0xDC143CFF, 0x00CED1FF, 0x9400D3FF, 0xFFD700FF,
];

const PALETTE_ADE20K_150: [[u8; 3]; 150] = [
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
];
