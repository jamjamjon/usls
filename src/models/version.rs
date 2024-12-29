#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Default)]
pub struct Version(pub u8, pub u8);

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = if self.1 == 0 {
            format!("v{}", self.0)
        } else {
            format!("v{}.{}", self.0, self.1)
        };
        write!(f, "{}", x)
    }
}

impl From<(u8, u8)> for Version {
    fn from((x, y): (u8, u8)) -> Self {
        Self(x, y)
    }
}

impl From<f32> for Version {
    fn from(x: f32) -> Self {
        let x = format!("{:?}", x);
        let x: Vec<u8> = x
            .as_str()
            .split('.')
            .map(|x| x.parse::<u8>().unwrap_or(0))
            .collect();
        Self(x[0], x[1])
    }
}

impl From<u8> for Version {
    fn from(x: u8) -> Self {
        Self(x, 0)
    }
}

impl Version {
    pub fn new(x: u8, y: u8) -> Self {
        Self(x, y)
    }
}
