use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::Dir;

/// Represents a downloadable asset in a release
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
}

/// Represents a GitHub release
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Release {
    pub tag_name: String,
    pub assets: Vec<Asset>,
}

/// Manages interactions with a GitHub repository's releases
pub struct Hub {
    /// github api
    _gh_api_release: String,

    /// GitHub repository owner
    owner: String,

    /// GitHub repository name          
    repo: String,

    /// Optional list of releases fetched from GitHub
    releases: Option<Vec<Release>>,

    /// Path to cache file
    cache: PathBuf,

    /// Optional release tag to be used
    tag: Option<String>,

    /// Filename for the asset, used in cache management
    file_name: Option<String>,
    file_size: Option<u64>,

    /// Full URL constructed for downloading the asset
    url: Option<String>,

    /// Local path where the asset will be stored
    path: PathBuf,

    /// Directory to store the downloaded file
    to: Dir,

    /// Download timeout in seconds
    timeout: u64,

    /// Time to live (cache duration)
    ttl: std::time::Duration,

    /// Maximum attempts for downloading
    max_attempts: u32,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hub")
            .field("owner", &self.owner)
            .field("repo", &self.repo)
            .field("cache", &self.cache)
            .field("path", &self.path)
            .field("releases", &self.releases.as_ref().map(|x| x.len()))
            .field("ttl", &self.ttl)
            .field("max_attempts", &self.max_attempts)
            .finish()
    }
}

impl Default for Hub {
    fn default() -> Self {
        let owner = "jamjamjon".to_string();
        let repo = "assets".to_string();
        let _gh_api_release = format!("https://api.github.com/repos/{}/{}/releases", owner, repo);

        Self {
            owner,
            repo,
            _gh_api_release,
            url: None,
            path: PathBuf::new(),
            to: Dir::Cache,
            tag: None,
            file_name: None,
            file_size: None,
            releases: None,
            cache: PathBuf::new(),
            timeout: 2000,
            max_attempts: 3,
            ttl: std::time::Duration::from_secs(10 * 60),
        }
    }
}

impl Hub {
    pub fn new() -> Result<Self> {
        let mut to = Dir::Cache;
        let cache = to
            .path(None)
            .or_else(|_| {
                to = Dir::Home;
                to.path(None)
            })?
            .join("cache_releases");

        Ok(Self {
            to,
            cache,
            ..Default::default()
        })
    }

    pub fn with_owner(mut self, owner: &str) -> Self {
        self.owner = owner.to_string();
        self
    }

    pub fn with_repo(mut self, repo: &str) -> Self {
        self.repo = repo.to_string();
        self
    }

    pub fn with_ttl(mut self, x: u64) -> Self {
        self.ttl = std::time::Duration::from_secs(x);
        self
    }

    pub fn with_timeout(mut self, x: u64) -> Self {
        self.timeout = x;
        self
    }

    pub fn with_max_attempts(mut self, x: u32) -> Self {
        self.max_attempts = x;
        self
    }

    pub fn fetch(mut self, s: &str) -> Result<Self> {
        let p = PathBuf::from(s);
        match p.exists() {
            true => self.path = p,
            false => {
                match s.split_once('/') {
                    Some((tag, file_name)) => {
                        // Extract tag and file from input string
                        self.tag = Some(tag.to_string());
                        self.file_name = Some(file_name.to_string());

                        // Check if releases are already loaded in memory
                        if self.releases.is_none() {
                            self.releases = Some(self.connect_remote()?);
                        }

                        if let Some(releases) = &self.releases {
                            // Validate the tag
                            let tags: Vec<&str> =
                                releases.iter().map(|x| x.tag_name.as_str()).collect();
                            if !tags.contains(&tag) {
                                anyhow::bail!(
                                    "Tag '{}' not found in releases. Available tags: {:?}",
                                    tag,
                                    tags
                                );
                            }

                            // Validate the file
                            if let Some(release) = releases.iter().find(|r| r.tag_name == tag) {
                                let files: Vec<&str> =
                                    release.assets.iter().map(|x| x.name.as_str()).collect();
                                if !files.contains(&file_name) {
                                    anyhow::bail!(
                                        "File '{}' not found in tag '{}'. Available files: {:?}",
                                        file_name,
                                        tag,
                                        files
                                    );
                                } else {
                                    for f_ in release.assets.iter() {
                                        if f_.name.as_str() == file_name {
                                            self.url = Some(f_.browser_download_url.clone());
                                            self.file_size = Some(f_.size);

                                            break;
                                        }
                                    }
                                }
                            }
                            self.path = self.to.path(Some(tag))?.join(file_name);
                        }
                    }
                    _ => anyhow::bail!(
                        "Download failed due to invalid format. Expected: <tag>/<file>, got: {}",
                        s
                    ),
                }
            }
        }

        Ok(self)
    }

    /// Fetch releases from GitHub and cache them
    fn fetch_and_cache_releases(url: &str, cache_path: &Path) -> Result<String> {
        let response = ureq::get(url)
            .set("User-Agent", "my-app")
            .call()
            .context("Failed to fetch releases from remote")?;

        if response.status() != 200 {
            anyhow::bail!(
                "Failed to fetch releases from remote ({}): status {} - {}",
                url,
                response.status(),
                response.status_text()
            );
        }

        let body = response
            .into_string()
            .context("Failed to read response body")?;

        // Ensure cache directory exists
        let parent_dir = cache_path
            .parent()
            .context("Invalid cache path; no parent directory found")?;
        std::fs::create_dir_all(parent_dir)
            .with_context(|| format!("Failed to create cache directory: {:?}", parent_dir))?;

        // Create temporary file
        let mut temp_file = tempfile::NamedTempFile::new_in(parent_dir)
            .context("Failed to create temporary cache file")?;

        // Write data to temporary file
        temp_file
            .write_all(body.as_bytes())
            .context("Failed to write to temporary cache file")?;

        // Persist temporary file as the cache
        temp_file.persist(cache_path).with_context(|| {
            format!("Failed to persist temporary cache file to {:?}", cache_path)
        })?;

        Ok(body)
    }

    pub fn tags(&mut self) -> Option<Vec<&str>> {
        if self.releases.is_none() {
            self.releases = self.connect_remote().ok();
        }

        self.releases
            .as_ref()
            .map(|releases| releases.iter().map(|x| x.tag_name.as_str()).collect())
    }

    pub fn files(&mut self, tag: &str) -> Option<Vec<&str>> {
        if self.releases.is_none() {
            self.releases = self.connect_remote().ok();
        }

        self.releases.as_ref().map(|releases| {
            releases
                .iter()
                .find(|r| r.tag_name == tag)
                .map(|a| a.assets.iter().map(|x| x.name.as_str()).collect())
        })?
    }

    pub fn connect_remote(&mut self) -> Result<Vec<Release>> {
        let should_download = if !self.cache.exists() {
            // println!("No cache found, fetching data from GitHub");
            true
        } else {
            match std::fs::metadata(&self.cache)?.modified() {
                Err(_) => {
                    // println!("Cannot get file modified time, fetching new data from GitHub");
                    true
                }
                Ok(modified_time) => {
                    if std::time::SystemTime::now().duration_since(modified_time)? < self.ttl {
                        // println!("Using cached data");
                        false
                    } else {
                        // println!("Cache expired, fetching new data from GitHub");
                        true
                    }
                }
            }
        };

        let body = if should_download {
            Self::fetch_and_cache_releases(&self._gh_api_release, &self.cache)?
        } else {
            std::fs::read_to_string(&self.cache)?
        };
        let releases: Vec<Release> = serde_json::from_str(&body)?;
        Ok(releases)
    }

    /// Commit the downloaded file, downloading if necessary
    pub fn commit(&self) -> Result<String> {
        if let Some(url) = &self.url {
            // Download if the file does not exist or if the size of file does not match
            if !self.path.is_file()
                || self.path.is_file()
                    && Some(std::fs::metadata(&self.path)?.len()) != self.file_size
            {
                let name = format!(
                    "{}/{}",
                    self.tag.as_ref().unwrap(),
                    self.file_name.as_ref().unwrap()
                );
                Self::download(
                    url.as_str(),
                    &self.path,
                    Some(&name),
                    Some(self.timeout),
                    Some(self.max_attempts),
                )?;
            }
        }
        self.path
            .to_str()
            .map(|s| s.to_string())
            .with_context(|| format!("Failed to convert PathBuf: {:?} to String", self.path))
    }

    /// Download a file from a given URL to a specified path with a progress bar
    pub fn download<P: AsRef<Path> + std::fmt::Debug>(
        src: &str,
        dst: P,
        prompt: Option<&str>,
        timeout: Option<u64>,
        max_attempts: Option<u32>,
    ) -> Result<()> {
        let max_attempts = max_attempts.unwrap_or(2);
        let timeout_duration = std::time::Duration::from_secs(timeout.unwrap_or(2000));
        let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();

        for i_try in 0..max_attempts {
            let resp = agent
                .get(src)
                .timeout(timeout_duration)
                .call()
                .with_context(|| {
                    format!(
                        "Failed to download file from {}, timeout: {:?}",
                        src, timeout_duration
                    )
                })?;
            let ntotal = resp
                .header("Content-Length")
                .and_then(|s| s.parse::<u64>().ok())
                .context("Content-Length header is missing or invalid")?;

            let pb = ProgressBar::new(ntotal);
            pb.set_style(
            ProgressStyle::with_template(
                "{prefix:.cyan.bold} {msg} |{bar}| ({percent_precise}%, {binary_bytes}/{binary_total_bytes}, {binary_bytes_per_sec})",
            )?
            .progress_chars("██ "),
        );
            pb.set_prefix(if i_try == 0 {
                "    Fetching"
            } else {
                " Re-Fetching"
            });
            pb.set_message(prompt.unwrap_or_default().to_string());

            let mut reader = resp.into_reader();
            let mut buffer = [0; 256];
            let mut downloaded_bytes = 0usize;
            let mut file = std::fs::File::create(&dst)
                .with_context(|| format!("Failed to create destination file: {:?}", dst))?;

            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                file.write_all(&buffer[..bytes_read])
                    .context("Failed to write to file")?;
                downloaded_bytes += bytes_read;
                pb.inc(bytes_read as u64);
            }

            // check size
            if downloaded_bytes as u64 != ntotal {
                continue;
            }

            // update
            pb.set_prefix("  Downloaded");
            pb.set_style(ProgressStyle::with_template(
                "{prefix:.green.bold} {msg} ({binary_total_bytes}) in {elapsed}",
            )?);
            pb.finish();

            if i_try != max_attempts {
                break;
            } else {
                anyhow::bail!("Exceeded the maximum number of download attempts");
            }
        }

        Ok(())
    }
}
