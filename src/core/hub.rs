use anyhow::{Context, Result};
use indicatif::ProgressStyle;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{retry, Dir};

/// Represents a downloadable asset in a release
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Asset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
}

/// Represents a GitHub release
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Release {
    pub tag_name: String,
    pub assets: Vec<Asset>,
}

// / Manages interactions with a GitHub repository's releases
/// Provides an interface for managing GitHub releases, including downloading assets,
/// fetching release tags and file information, and handling caching.
///
/// The `Hub` struct simplifies interactions with a GitHub repository by allowing users
/// to specify a repository owner and name, download files from releases, and manage
/// cached data to reduce redundant network requests.
///
/// # Fields
/// - `owner`: The owner of the GitHub repository (e.g., `"jamjamjon"`).
/// - `repo`: The name of the GitHub repository (e.g., `"assets"`).
/// - `to`: The directory where downloaded files are stored, determined from a prioritized list
///   of available directories (e.g., cache, home, config, or current directory).
/// - `timeout`: Timeout duration for network requests, in seconds.
/// - `ttl`: Time-to-live duration for cached data, defining how long cache files remain valid.
/// - `max_attempts`: The maximum number of retry attempts for failed downloads or network operations.
///
/// # Example
///
/// ## 1. Download from a default GitHub release
/// Download a file by specifying its path relative to the release:
/// ```rust,ignore
/// let path = Hub::default().try_fetch("images/bus.jpg")?;
/// println!("Fetched image to: {:?}", path);
/// ```
///
/// ## 2. Download from a specific GitHub release URL
/// Fetch a file directly using its full GitHub release URL:
/// ```rust,ignore
/// let path = Hub::default()
///     .try_fetch("https://github.com/jamjamjon/assets/releases/download/images/bus.jpg")?;
/// println!("Fetched file to: {:?}", path);
/// ```
///
/// ## 3. Fetch available tags and files in a repository
/// List all release tags and the files associated with each tag:
/// ```rust,ignore
/// let hub = Hub::default().with_owner("jamjamjon").with_repo("usls");
/// for tag in hub.tags().iter() {
///     let files = hub.files(tag);
///     println!("Tag: {}, Files: {:?}", tag, files);
/// }
/// ```
///
/// # Default Behavior
/// By default, `Hub` interacts with the `jamjamjon/assets` repository, stores downloads in
/// an accessible directory, and applies a 10-minute cache expiration time. These settings
/// can be customized using the builder-like methods `with_owner`, `with_repo`, `with_ttl`,
/// `with_timeout`, and `with_max_attempts`.
///
/// # Errors
/// Methods in `Hub` return `Result` types. Errors may occur due to invalid paths, failed
/// network requests, cache write failures, or mismatched file sizes during downloads.
///
#[derive(Debug)]
pub struct Hub {
    /// GitHub repository owner
    owner: String,

    /// GitHub repository name          
    repo: String,

    /// Directory to store the downloaded file
    to: Dir,

    /// Time to live (cache duration)
    ttl: Duration,

    /// The maximum number of retry attempts for failed downloads or network operations
    max_attempts: u32,
}

impl Default for Hub {
    fn default() -> Self {
        let owner = "jamjamjon".to_string();
        let repo = "assets".to_string();
        let to = [Dir::Cache, Dir::Home, Dir::Config, Dir::Current]
            .into_iter()
            .find(|dir| dir.crate_dir_default().is_ok())
            .expect(
                "Unable to get cache directory, home directory, config directory, and current directory. Possible reason: \
                \n1. Unsupported OS \
                \n2. Directory does not exist \
                \n3. Insufficient permissions to access"
            );

        Self {
            owner,
            repo,
            to,
            max_attempts: 3,
            ttl: Duration::from_secs(10 * 60),
        }
    }
}

impl Hub {
    pub fn new(owner: &str, repo: &str) -> Self {
        Self {
            owner: owner.into(),
            repo: repo.into(),
            ..Default::default()
        }
    }

    /// Attempts to fetch a file from a local path or a GitHub release.
    ///
    /// The `try_fetch` method supports three main scenarios:
    /// 1. **Local file**: If the provided string is a valid file path, the file is returned without downloading.
    /// 2. **GitHub release URL**: If the input matches a valid GitHub release URL, the corresponding file is downloaded.
    /// 3. **Default repository**: If no explicit URL is provided, the method uses the default or configured repository.
    ///
    /// # Parameters
    /// - `s`: A string representing the file to fetch. This can be:
    ///   - A local file path.
    ///   - A GitHub release URL (e.g., `https://github.com/owner/repo/releases/download/tag/file`).
    ///   - A `<tag>/<file>` format for fetching from the default repository.
    ///
    /// # Returns
    /// - `Result<String>`: On success, returns the path to the fetched file.
    ///
    /// # Errors
    /// - Returns an error if:
    ///   - The file cannot be found locally.
    ///   - The URL or tag is invalid.
    ///   - Network operations fail after the maximum retry attempts.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut hub = Hub::default();
    ///
    /// // Fetch a file from a local path
    /// let local_path = hub.try_fetch("local/path/to/file").expect("File not found");
    ///
    /// // Fetch a file from a GitHub release URL
    /// let url_path = hub.try_fetch("https://github.com/owner/repo/releases/download/tag/file")
    ///     .expect("Failed to fetch file");
    ///
    /// // Fetch a file using the default repository
    /// let default_repo_path = hub.try_fetch("v1.0.0/file").expect("Failed to fetch file");
    /// ```
    pub fn try_fetch(&mut self, s: &str) -> Result<String> {
        #[derive(Default, Debug, aksr::Builder)]
        struct Pack {
            // owner: String,
            // repo: String,
            url: String,
            tag: String,
            file_name: String,
            file_size: Option<u64>,
        }
        let mut pack = Pack::default();

        // saveout
        let p = PathBuf::from(s);
        let saveout = if p.exists() {
            // => Local file
            p
        } else if let Some((owner_, repo_, tag_, file_name_)) = Self::is_valid_github_release_url(s)
        {
            // => Valid GitHub release URL
            // keep original owner, repo and tag
            let saveout = self
                .to
                .crate_dir_default_with_subs(&[&owner_, &repo_, &tag_])?
                .join(&file_name_);

            pack = pack.with_url(s).with_tag(&tag_).with_file_name(&file_name_);
            if let Some(n) = retry!(self.max_attempts, self.fetch_get_response(s))?
                .headers()
                .get(http::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
            {
                pack = pack.with_file_size(n);
            }

            saveout
        } else {
            // => Default hub

            // Fetch releases
            let releases = match self.get_releases(&self.owner, &self.repo, &self.to, &self.ttl) {
                Err(err) => anyhow::bail!(
                    "Failed to download: No releases found in this repo. Error: {}",
                    err
                ),
                Ok(releases) => releases,
            };

            // Check remote
            match s.split_once('/') {
                Some((tag_, file_name_)) => {
                    // Validate the tag
                    let tags: Vec<String> = releases.iter().map(|x| x.tag_name.clone()).collect();
                    if !tags.contains(&tag_.to_string()) {
                        anyhow::bail!(
                            "Failed to download: Tag `{}` not found in GitHub releases. Available tags: {:?}",
                            tag_,
                            tags
                        );
                    } else {
                        // Validate the file
                        if let Some(release) = releases.iter().find(|r| r.tag_name == tag_) {
                            let files: Vec<&str> =
                                release.assets.iter().map(|x| x.name.as_str()).collect();
                            if !files.contains(&file_name_) {
                                anyhow::bail!(
                                    "Failed to download: The file `{}` is missing in tag `{}`. Available files: {:?}",
                                    file_name_,
                                    tag_,
                                    files
                                );
                            } else {
                                for f_ in release.assets.iter() {
                                    if f_.name.as_str() == file_name_ {
                                        pack = pack
                                            .with_url(&f_.browser_download_url)
                                            .with_tag(tag_)
                                            .with_file_name(file_name_)
                                            .with_file_size(f_.size);
                                        break;
                                    }
                                }
                            }
                        }

                        self.to
                            .crate_dir_default_with_subs(&[tag_])?
                            .join(file_name_)
                    }
                }
                _ => anyhow::bail!(
                    "Failed to download file from github releases due to invalid format. Expected: <tag>/<file>, got: {}",
                    s
                ),
            }
        };

        // Commit the downloaded file, downloading if necessary
        if !pack.url.is_empty() {
            // Download if the file does not exist or if the size of file does not match
            if saveout.is_file() {
                match pack.file_size {
                    None => {
                        log::warn!(
                            "Failed to retrieve the remote file size. \
                            Download will be skipped, which may cause issues. \
                            Please verify your network connection or ensure the local file is valid and complete."
                        );
                    }
                    Some(file_size) => {
                        if std::fs::metadata(&saveout)?.len() != file_size {
                            log::debug!(
                                "Local file size does not match remote. Starting download."
                            );
                            retry!(
                                self.max_attempts,
                                1000,
                                3000,
                                self.download(
                                    &pack.url,
                                    &saveout,
                                    Some(&format!("{}/{}", pack.tag, pack.file_name)),
                                )
                            )?;
                        } else {
                            log::debug!("Local file size matches remote. No download required.");
                        }
                    }
                }
            } else {
                log::debug!("Starting remote file download...");
                retry!(
                    self.max_attempts,
                    1000,
                    3000,
                    self.download(
                        &pack.url,
                        &saveout,
                        Some(&format!("{}/{}", pack.tag, pack.file_name)),
                    )
                )?;
            }
        }

        saveout
            .to_str()
            .map(|s| s.to_string())
            .with_context(|| format!("Failed to convert PathBuf: {:?} to String", saveout))
    }

    /// Fetch releases from GitHub and cache them
    fn fetch_and_cache_releases(&self, url: &str, cache_path: &Path) -> Result<String> {
        let response = retry!(self.max_attempts, self.fetch_get_response(url))?;
        let body = response
            .into_body()
            .read_to_string()
            .context("Failed to read response body")?;

        // Ensure cache directory exists
        let parent_dir = cache_path
            .parent()
            .context("Invalid cache path: no parent directory found")?;
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

    pub fn tags(&self) -> Vec<String> {
        self.get_releases(&self.owner, &self.repo, &self.to, &self.ttl)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x.tag_name)
            .collect()
    }

    pub fn files(&self, tag: &str) -> Vec<String> {
        self.get_releases(&self.owner, &self.repo, &self.to, &self.ttl)
            .unwrap_or_default()
            .into_iter()
            .find(|r| r.tag_name == tag)
            .map(|a| a.assets.iter().map(|x| x.name.clone()).collect())
            .unwrap_or_default()
    }

    pub fn is_file_expired<P: AsRef<Path>>(file: P, ttl: &Duration) -> Result<bool> {
        let file = file.as_ref();
        let y = if !file.exists() {
            log::debug!("No cache found, fetching data from GitHub");
            true
        } else {
            match std::fs::metadata(file)?.modified() {
                Err(_) => {
                    log::debug!("Cannot get file modified time, fetching new data from GitHub");
                    true
                }
                Ok(modified_time) => {
                    if std::time::SystemTime::now().duration_since(modified_time)? < *ttl {
                        log::debug!("Using cached data");
                        false
                    } else {
                        log::debug!("Cache expired, fetching new data from GitHub");
                        true
                    }
                }
            }
        };
        Ok(y)
    }

    /// Download a file from a github release to a specified path with a progress bar
    pub fn download<P: AsRef<Path> + std::fmt::Debug>(
        &self,
        src: &str,
        dst: P,
        message: Option<&str>,
    ) -> Result<()> {
        let resp = self.fetch_get_response(src)?;
        let ntotal = resp
            .headers()
            .get(http::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
            .context("Content-Length header is missing or invalid")?;
        let pb = crate::build_progress_bar(
            ntotal,
            "Fetching",
            Some(message.unwrap_or_default()),
            "{prefix:.cyan.bold} {msg} |{bar}| ({percent_precise}%, {binary_bytes}/{binary_total_bytes}, {binary_bytes_per_sec})",
        )?;

        let mut reader = resp.into_body().into_reader();
        const BUFFER_SIZE: usize = 64 * 1024;
        let mut buffer = [0; BUFFER_SIZE];
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

        if downloaded_bytes as u64 != ntotal {
            anyhow::bail!("The downloaded file is incomplete.");
        }

        // Update the progress bar
        pb.set_prefix("Downloaded");
        pb.set_style(ProgressStyle::with_template(
            crate::PROGRESS_BAR_STYLE_FINISH_3,
        )?);
        pb.finish();

        Ok(())
    }

    fn fetch_get_response(&self, url: &str) -> anyhow::Result<http::Response<ureq::Body>> {
        let config = ureq::Agent::config_builder()
            .proxy(ureq::Proxy::try_from_env())
            .build();
        let agent = ureq::Agent::new_with_config(config);

        let response = agent
            .get(url)
            .call()
            .map_err(|err| anyhow::anyhow!("Failed to GET response from {}: {}", url, err))?;
        if response.status() != 200 {
            anyhow::bail!("Failed to fetch data from remote due to: {:?}", response);
        }

        Ok(response)
    }

    fn cache_file(owner: &str, repo: &str) -> String {
        let safe_owner = owner.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let safe_repo = repo.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        format!("releases-{}-{}.json", safe_owner, safe_repo)
    }

    fn get_releases(
        &self,
        owner: &str,
        repo: &str,
        to: &Dir,
        ttl: &Duration,
    ) -> Result<Vec<Release>> {
        let cache = to
            .crate_dir_default_with_subs(&["caches"])?
            .join(Self::cache_file(owner, repo));
        let is_file_expired = Self::is_file_expired(&cache, ttl)?;
        let body = if is_file_expired {
            let gh_api_release = format!(
                "https://api.github.com/repos/{}/{}/releases?per_page=100",
                owner, repo
            );
            self.fetch_and_cache_releases(&gh_api_release, &cache)?
        } else {
            std::fs::read_to_string(&cache)?
        };

        Ok(serde_json::from_str(&body)?)
    }

    pub fn is_valid_github_release_url(url: &str) -> Option<(String, String, String, String)> {
        let re =
            Regex::new(r"^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)$")
                .expect("Failed to compile the regex for GitHub release URL pattern");

        if let Some(caps) = re.captures(url) {
            let owner = caps.get(1).map_or("", |m| m.as_str());
            let repo = caps.get(2).map_or("", |m| m.as_str());
            let tag = caps.get(3).map_or("", |m| m.as_str());
            let file = caps.get(4).map_or("", |m| m.as_str());

            Some((
                owner.to_string(),
                repo.to_string(),
                tag.to_string(),
                file.to_string(),
            ))
        } else {
            None
        }
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

    pub fn with_max_attempts(mut self, x: u32) -> Self {
        self.max_attempts = x;
        self
    }
}
