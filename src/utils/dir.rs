use anyhow::Result;
use std::path::PathBuf;

/// Represents various directories on the system, including Home, Cache, Config, and more.
#[derive(Debug)]
pub enum Dir {
    Home,
    Cache,
    Config,
    Current,
}

impl Dir {
    /// Returns the raw path for the directory without adding the `crate_name` subdirectory.
    ///
    /// Examples:
    /// `~/.cache`, `~/.config`, `~`.
    ///
    pub fn base_dir(&self) -> Result<PathBuf> {
        let p = match self {
            Dir::Home => dirs::home_dir(),
            Dir::Cache => dirs::cache_dir(),
            Dir::Config => dirs::config_dir(),
            Dir::Current => std::env::current_dir().ok(),
        };

        let  p = p.ok_or_else(|| {
            anyhow::anyhow!("Failed to retrieve base path for {:?}. Unsupported operating system. Now supports Linux, MacOS, Windows.", self)
        })?;

        Ok(p)
    }

    /// Returns the default path for the `crate_name` directory, creating it automatically if it does not exist.
    ///
    /// Examples:
    /// `~/.cache/crate_name`, `~/.config/crate_name`, `~/.crate_name`.
    pub fn crate_dir(&self, crate_name: &str) -> Result<PathBuf> {
        let mut p = self.base_dir()?;

        if let Dir::Home = self {
            p.push(format!(".{}", crate_name));
        } else {
            p.push(crate_name);
        }

        self.try_create_directory(&p)?;

        Ok(p)
    }

    /// Returns the default path for the `usls` directory, creating it automatically if it does not exist.
    ///
    /// Examples:
    /// `~/.cache/usls`, `~/.config/usls`, `~/.usls`.
    pub fn crate_dir_default(&self) -> Result<PathBuf> {
        self.crate_dir(crate::CRATE_NAME)
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
    pub fn base_dir_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.base_dir()?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    pub fn base_dir_with_filename(&self, filename: &str) -> anyhow::Result<std::path::PathBuf> {
        let d = self.base_dir()?.join(filename);
        self.try_create_directory(&d)?;
        Ok(d)
    }

    /// Constructs a path to the `crate_name` directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/crate_name/sub1/sub2/sub3`, `~/.config/crate_name/sub1/sub2`, `~/.crate_name/sub1/sub2`.
    pub fn crate_dir_with_subs(
        &self,
        crate_name: &str,
        subs: &[&str],
    ) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.crate_dir(crate_name)?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    /// Constructs a path to the `usls` directory with the provided subdirectories, creating it automatically.
    ///
    /// Examples:
    /// `~/.cache/usls/sub1/sub2/sub3`, `~/.config/usls/sub1/sub2`, `~/.usls/sub1/sub2`.
    pub fn crate_dir_default_with_subs(&self, subs: &[&str]) -> anyhow::Result<std::path::PathBuf> {
        let mut d = self.crate_dir_default()?;
        self.append_subs(&mut d, subs)?;
        Ok(d)
    }

    pub fn crate_dir_with_filename(
        &self,
        crate_name: &str,
        filename: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let d = self.crate_dir(crate_name)?.join(filename);
        self.try_create_directory(&d)?;
        Ok(d)
    }

    pub fn crate_dir_default_with_filename(
        &self,
        filename: &str,
    ) -> anyhow::Result<std::path::PathBuf> {
        let d = self.crate_dir_default()?.join(filename);
        self.try_create_directory(&d)?;
        Ok(d)
    }

    /// Appends subdirectories to the given base path and creates the directories if they don't exist.
    fn append_subs(&self, path: &mut std::path::PathBuf, subs: &[&str]) -> anyhow::Result<()> {
        for sub in subs {
            path.push(sub);
        }
        self.try_create_directory(path)?;
        Ok(())
    }

    fn try_create_directory<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Err(err) = std::fs::create_dir_all(path) {
            return Err(anyhow::anyhow!(
                "Failed to create directory at {:?}: {}",
                path,
                err
            ));
        }
        Ok(())
    }
}
