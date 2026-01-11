use std::str::FromStr;

/// Model scale variants for different model sizes.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Scale {
    N,
    T,
    B,
    S,
    M,
    L,
    C,
    E,
    X,
    G,
    P,
    A,
    F,
    Million(f32),
    Billion(f32),
    Named(String),
}

impl std::fmt::Display for Scale {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::N => write!(f, "n"),
            Self::T => write!(f, "t"),
            Self::S => write!(f, "s"),
            Self::B => write!(f, "b"),
            Self::M => write!(f, "m"),
            Self::L => write!(f, "l"),
            Self::C => write!(f, "c"),
            Self::E => write!(f, "e"),
            Self::X => write!(f, "x"),
            Self::G => write!(f, "g"),
            Self::P => write!(f, "p"),
            Self::A => write!(f, "a"),
            Self::F => write!(f, "f"),
            Self::Million(x) => write!(f, "{}m", x),
            Self::Billion(x) => write!(f, "{}b", x),
            Scale::Named(x) => write!(f, "{}", x),
        }
    }
}

impl TryFrom<char> for Scale {
    type Error = anyhow::Error;

    fn try_from(s: char) -> Result<Self, Self::Error> {
        match s.to_ascii_lowercase() {
            'n' => Ok(Self::N),
            't' => Ok(Self::T),
            'b' => Ok(Self::B),
            's' => Ok(Self::S),
            'm' => Ok(Self::M),
            'l' => Ok(Self::L),
            'c' => Ok(Self::C),
            'e' => Ok(Self::E),
            'x' => Ok(Self::X),
            'g' => Ok(Self::G),
            'p' => Ok(Self::P),
            'a' => Ok(Self::A),
            'f' => Ok(Self::F),
            x => anyhow::bail!("Unsupported model scale: {:?}", x),
        }
    }
}

impl FromStr for Scale {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "n" | "nano" => Ok(Self::N),
            "t" | "tiny" => Ok(Self::T),
            "b" | "base" => Ok(Self::B),
            "s" | "small" => Ok(Self::S),
            "m" | "medium" => Ok(Self::M),
            "l" | "large" => Ok(Self::L),
            "c" => Ok(Self::C),
            "e" => Ok(Self::E),
            "x" | "extra-large" => Ok(Self::X),
            "g" | "giant" => Ok(Self::G),
            "p" | "pico" => Ok(Self::P),
            "a" | "atto" => Ok(Self::A),
            "f" | "femto" => Ok(Self::F),
            scale if scale.ends_with("b") => {
                let num_str = &scale[..scale.len() - 1];
                match num_str.parse::<f32>() {
                    Ok(x) => Ok(Self::Billion(x)),
                    Err(_) => Ok(Self::Named(s.to_string())),
                }
            }
            scale if scale.ends_with("m") => {
                let num_str = &scale[..scale.len() - 1];
                match num_str.parse::<f32>() {
                    Ok(x) => Ok(Self::Million(x)),
                    Err(_) => Ok(Self::Named(s.to_string())),
                }
            }
            _ => Ok(Self::Named(s.to_string())),
        }
    }
}
