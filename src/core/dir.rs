/// Represents various directories on the system, including Home, Cache, Config, and more.
#[derive(Debug)]
pub enum Dir {
    Home,
    Cache,
    Config,
    Currnet,
    Document,
    Data,
    Download,
    Desktop,
    Audio,
    Picture,
}

impl Dir {
    pub fn saveout(subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        Self::Currnet.raw_path_with_subs(subs)
    }

    /// Retrieves the base path for the specified directory type, optionally appending the `usls` subdirectory.
    ///
    /// # Arguments
    /// * `raw` - If `true`, returns the base path without adding the `usls` subdirectory.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The base path for the directory.
    fn get_path(&self, raw: bool) -> anyhow::Result<std::path::PathBuf> {
        let base_path = match self {
            Dir::Home => dirs::home_dir(),
            Dir::Cache => dirs::cache_dir(),
            Dir::Config => dirs::config_dir(),
            Dir::Currnet => std::env::current_dir().ok(),
            _ => None,
        };

        let mut path = base_path.ok_or_else(|| {
            anyhow::anyhow!("Unsupported operating system. Now supports Linux, MacOS, Windows.")
        })?;

        if !raw {
            if let Dir::Home = self {
                path.push(".usls");
            } else {
                path.push("usls");
            }
        }
        Ok(path)
    }

    /// Returns the default path for the `usls` directory, creating it automatically if it does not exist.
    ///
    /// Examples:
    /// `~/.cache/usls`, `~/.config/usls`, `~/.usls`.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The default `usls` directory path.
    pub fn path(&self) -> anyhow::Result<std::path::PathBuf> {
        let d = self.get_path(false)?;
        self.create_directory(&d)?;
        Ok(d)
    }

    /// Returns the raw path for the directory without adding the `usls` subdirectory.
    ///
    /// Examples:
    /// `~/.cache`, `~/.config`, `~`.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The raw directory path.
    pub fn raw_path(&self) -> anyhow::Result<std::path::PathBuf> {
        self.get_path(true)
    }

    /// Constructs a path to the `usls` directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/usls/sub1/sub2/sub3`, `~/.config/usls/sub1/sub2`, `~/.usls/sub1/sub2`.
    ///
    /// # Arguments
    /// * `subs` - A slice of strings representing subdirectories to append.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The resulting directory path.
    pub fn path_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.get_path(false)?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    /// Constructs a path to a specified directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/sub1/sub2/sub3`, `~/.config/sub1/sub2`, `~/sub1/sub2`.
    ///
    /// # Arguments
    /// * `subs` - A slice of strings representing subdirectories to append.
    ///
    /// # Returns
    /// * `Result<PathBuf>` - The resulting directory path.
    pub fn raw_path_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.get_path(true)?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    /// Appends subdirectories to the given base path and creates the directories if they don't exist.
    fn append_subs(&self, path: &mut std::path::PathBuf, subs: &[&str]) -> anyhow::Result<()> {
        for sub in subs {
            path.push(sub);
        }
        self.create_directory(path)?;
        Ok(())
    }

    /// Creates the specified directory if it does not exist.
    fn create_directory(&self, path: &std::path::PathBuf) -> anyhow::Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }
}
