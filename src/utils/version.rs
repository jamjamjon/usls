use aksr::Builder;
use std::convert::TryFrom;

/// Version representation with major, minor, and optional patch numbers.
#[derive(Debug, Builder, PartialEq, Eq, Copy, Clone, Hash, Default, PartialOrd, Ord)]
pub struct Version(pub u8, pub u8, pub Option<u8>);

impl Version {
    pub fn new(major: u8, minor: u8) -> Self {
        Self(major, minor, None)
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.2 {
            None => {
                if self.1 == 0 {
                    write!(f, "v{}", self.0)
                } else {
                    write!(f, "v{}.{}", self.0, self.1)
                }
            }
            Some(patch) => write!(f, "v{}.{}.{}", self.0, self.1, patch),
        }
    }
}

impl From<u8> for Version {
    fn from(major: u8) -> Self {
        Self(major, 0, None)
    }
}

impl From<(u8, u8)> for Version {
    fn from((major, minor): (u8, u8)) -> Self {
        Self(major, minor, None)
    }
}

impl From<(u8, u8, u8)> for Version {
    fn from((major, minor, patch): (u8, u8, u8)) -> Self {
        Self(major, minor, Some(patch))
    }
}

impl TryFrom<f32> for Version {
    type Error = anyhow::Error;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !(0.0..=255.0).contains(&value) {
            return Err(anyhow::anyhow!("Float value out of range for Version"));
        }

        let major = value.trunc() as u8;
        let minor_float = (value - major as f32) * 10.0;
        let minor = minor_float.trunc() as u8;

        Ok(Self(major, minor, None))
    }
}
