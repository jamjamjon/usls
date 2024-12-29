use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::{Dir, PREFIX_LENGTH};

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
    /// GitHub repository owner
    owner: String,

    /// GitHub repository name          
    repo: String,

    /// Directory to store the downloaded file
    to: Dir,

    /// Path to cache file
    cache: PathBuf,

    /// Optional list of releases fetched from GitHub
    releases: Vec<Release>,

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
            // .field("releases", &self.releases.as_ref().map(|x| x.len()))
            .field("ttl", &self.ttl)
            .field("max_attempts", &self.max_attempts)
            .finish()
    }
}

impl Hub {
    pub fn new() -> Result<Self> {
        // Build the Hub instance
        let owner = "jamjamjon".to_string();
        let repo = "assets".to_string();
        let mut to = Dir::Cache;
        let cache = to
            .path()
            .or_else(|_| {
                to = Dir::Home;
                to.path()
            })
            .or_else(|_| {
                to = Dir::Config;
                to.path()
            })
            .or_else(|_| {
                to = Dir::Current;
                to.path()
            })
            .expect(
                "Unable to get cache directory, home directory, config directory, and current directory. Possible reason:\
                \n1. Unsupported OS\
                \n2. Directory does not exist\
                \n3. Insufficient permissions to access"
            )
            .join(".gh_releases.cache");

        let ttl = std::time::Duration::from_secs(10 * 60);

        // releases
        let is_file_expired = Self::is_file_expired(&cache, ttl)?;
        let body = if is_file_expired {
            let gh_api_release =
                format!("https://api.github.com/repos/{}/{}/releases", owner, repo);
            Self::fetch_and_cache_releases(&gh_api_release, &cache)?
        } else {
            std::fs::read_to_string(&cache)?
        };
        let releases = serde_json::from_str(&body)?;

        Ok(Self {
            owner,
            repo,
            to,
            cache,
            releases,
            ttl,
            timeout: 3000,
            max_attempts: 3,
        })
    }

    pub fn try_fetch(&mut self, s: &str) -> Result<String> {
        // mutables
        let mut url: Option<String> = None;
        let mut tag: Option<String> = None;
        let mut file_size: Option<u64> = None;
        let mut file_name: Option<String> = None;

        let p = PathBuf::from(s);
        let path = match p.exists() {
            true => p,
            false => {
                // check empty
                if self.releases.is_empty() {
                    anyhow::bail!("No releases found in this repo.");
                }

                // check remote
                match s.split_once('/') {
                    Some((tag_, file_name_)) => {
                        // Extract tag and file from input string
                        tag = Some(tag_.to_string());
                        file_name = Some(file_name_.to_string());

                        // Validate the tag
                        let tags = self.tags();
                        if !tags.contains(&tag_) {
                            anyhow::bail!(
                                    "Try to fetch from GitHub releases. However, tag: `{}` is not found. Available tags: {:#?}",
                                    tag_,
                                    tags
                                );
                        }

                        // Validate the file
                        if let Some(release) = self.releases.iter().find(|r| r.tag_name == tag_) {
                            let files: Vec<&str> =
                                release.assets.iter().map(|x| x.name.as_str()).collect();
                            if !files.contains(&file_name_) {
                                anyhow::bail!(
                                            "Try to fetch from GitHub releases. However, file: `{}` is not found in tag: `{}`. Available files: {:#?}",
                                            file_name_,
                                            tag_,
                                            files
                                        );
                            } else {
                                for f_ in release.assets.iter() {
                                    if f_.name.as_str() == file_name_ {
                                        url = Some(f_.browser_download_url.clone());
                                        file_size = Some(f_.size);

                                        break;
                                    }
                                }
                            }
                        }
                        self.to.path_with_subs(&[tag_])?.join(file_name_)
                    }
                    _ => anyhow::bail!(
                        "Download failed due to invalid format. Expected: <tag>/<file>, got: {}",
                        s
                    ),
                }
            }
        };

        // Commit the downloaded file, downloading if necessary
        if let Some(url) = &url {
            // Download if the file does not exist or if the size of file does not match
            if !path.is_file()
                || path.is_file() && Some(std::fs::metadata(&path)?.len()) != file_size
            {
                let name = format!("{}/{}", tag.as_ref().unwrap(), file_name.as_ref().unwrap());
                Self::download(
                    url.as_str(),
                    &path,
                    Some(&name),
                    Some(self.timeout),
                    Some(self.max_attempts),
                )?;
            }
        }

        path.to_str()
            .map(|s| s.to_string())
            .with_context(|| format!("Failed to convert PathBuf: {:?} to String", path))
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

        // Encode?

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

    pub fn tags(&self) -> Vec<&str> {
        self.releases.iter().map(|x| x.tag_name.as_str()).collect()
    }

    pub fn files(&self, tag: &str) -> Vec<&str> {
        self.releases
            .iter()
            .find(|r| r.tag_name == tag)
            .map(|a| a.assets.iter().map(|x| x.name.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn is_file_expired<P: AsRef<Path>>(file: P, ttl: std::time::Duration) -> Result<bool> {
        let file = file.as_ref();
        let y = if !file.exists() {
            debug!("No cache found, fetching data from GitHub");
            true
        } else {
            match std::fs::metadata(file)?.modified() {
                Err(_) => {
                    debug!("Cannot get file modified time, fetching new data from GitHub");
                    true
                }
                Ok(modified_time) => {
                    if std::time::SystemTime::now().duration_since(modified_time)? < ttl {
                        debug!("Using cached data");
                        false
                    } else {
                        debug!("Cache expired, fetching new data from GitHub");
                        true
                    }
                }
            }
        };
        Ok(y)
    }

    /// Download a file from a github release to a specified path with a progress bar
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
                format!("{:>PREFIX_LENGTH$}", "Fetching")
            } else {
                format!("{:>PREFIX_LENGTH$}", "Re-Fetching")
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
            pb.set_prefix("Downloaded");
            pb.set_style(ProgressStyle::with_template(
                crate::PROGRESS_BAR_STYLE_FINISH_3,
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
}
