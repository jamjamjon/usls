#[derive(Debug)]
pub enum Dir {
    Home,
    Cache,
    Config,
    Document,
    Data,
    Download,
    Desktop,
    Audio,
    Picture,
}

impl Dir {
    pub fn path(&self, sub: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
        let mut d = match self {
            Dir::Home => match dirs::home_dir() {
                Some(mut d) => {
                    d.push(".usls");
                    d
                }
                None => anyhow::bail!(
                    "Unsupported operating system. Now support Linux, MacOS, Windows."
                ),
            },
            Dir::Cache => match dirs::cache_dir() {
                Some(mut d) => {
                    d.push("usls");
                    d
                }
                None => anyhow::bail!(
                    "Unsupported operating system. Now support Linux, MacOS, Windows."
                ),
            },
            Dir::Config => match dirs::config_dir() {
                Some(mut d) => {
                    d.push("usls");
                    d
                }
                None => anyhow::bail!(
                    "Unsupported operating system. Now support Linux, MacOS, Windows."
                ),
            },
            _ => todo!(),
        };
        if let Some(sub) = sub {
            d.push(sub);
        }
        if !d.exists() {
            std::fs::create_dir_all(&d).expect("Failed to create usls config directory.");
        }
        Ok(d)
    }
}
