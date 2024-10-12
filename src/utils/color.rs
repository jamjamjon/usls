#[derive(Copy, Clone)]
pub struct Color(u32);

impl std::fmt::Debug for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Color")
            .field("RGBA", &self.rgba())
            .field("HEX", &self.hex())
            .finish()
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

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

impl TryFrom<&str> for Color {
    type Error = &'static str;

    fn try_from(x: &str) -> Result<Self, Self::Error> {
        let hex = x.trim_start_matches('#');
        let hex = match hex.len() {
            6 => format!("{}ff", hex),
            8 => hex.to_string(),
            _ => return Err("Failed to convert `Color` from str: invalid length"),
        };

        u32::from_str_radix(&hex, 16)
            .map(Self)
            .map_err(|_| "Failed to convert `Color` from str: invalid hex")
    }
}

impl Color {
    const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32))
    }

    pub fn rgba(&self) -> (u8, u8, u8, u8) {
        let r = ((self.0 >> 24) & 0xff) as u8;
        let g = ((self.0 >> 16) & 0xff) as u8;
        let b = ((self.0 >> 8) & 0xff) as u8;
        let a = (self.0 & 0xff) as u8;
        (r, g, b, a)
    }

    pub fn rgb(&self) -> (u8, u8, u8) {
        let (r, g, b, _) = self.rgba();
        (r, g, b)
    }

    pub fn bgr(&self) -> (u8, u8, u8) {
        let (r, g, b) = self.rgb();
        (b, g, r)
    }

    pub fn hex(&self) -> String {
        format!("#{:08x}", self.0)
    }

    pub fn create_palette<A: Into<Self> + Copy>(xs: &[A]) -> Vec<Self> {
        xs.iter().copied().map(Into::into).collect()
    }

    pub fn palette1() -> Vec<Self> {
        // TODO
        Self::create_palette(&[
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
            0x708090ff, // SlateGray
            0x7fffd4ff, // AquaMarine
            0x3399ffff, // Blue2
            0x00ffffff, // Cyan
            0x8a2befff, // BlueViolet
            0xa52a2aff, // Brown
            0xd8bfd8ff, // Thistle
            0xf0ffffff, // Azure
            0x609ea0ff, // CadetBlue
        ])
    }

    pub fn palette2() -> Vec<Self> {
        // TODO
        Self::create_palette(&[
            0x00202eff, 0x003f5cff, 0x2c4875ff, 0x8a508fff, 0xbc5090ff, 0xff6361ff, 0xff8531ff,
            0xffa600ff, 0xffd380ff,
        ])
    }
}
