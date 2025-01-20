#[derive(Debug, Copy, Clone)]
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
}

impl std::fmt::Display for Scale {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let x = match self {
            Self::N => "n",
            Self::T => "t",
            Self::S => "s",
            Self::B => "b",
            Self::M => "m",
            Self::L => "l",
            Self::C => "c",
            Self::E => "e",
            Self::X => "x",
            Self::G => "g",
            Self::P => "p",
            Self::A => "a",
            Self::F => "f",
            Self::Million(x) => &format!("{x}m"),
            Self::Billion(x) => &format!("{x}b"), // x.0 -> x
        };
        write!(f, "{}", x)
    }
}

impl TryFrom<char> for Scale {
    type Error = anyhow::Error;

    fn try_from(s: char) -> Result<Self, Self::Error> {
        match s {
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

impl TryFrom<&str> for Scale {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
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
                    Err(_) => anyhow::bail!("Invalid Billion format: {}", scale),
                }
            }
            scale if scale.ends_with("m") => {
                let num_str = &scale[..scale.len() - 1];
                match num_str.parse::<f32>() {
                    Ok(x) => Ok(Self::Million(x)),
                    Err(_) => anyhow::bail!("Invalid Million format: {}", scale),
                }
            }
            x => anyhow::bail!("Unsupported model scale: {:?}", x),
        }
    }
}
