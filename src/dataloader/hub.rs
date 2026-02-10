use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::NamedTempFile;

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

/// Manages downloading and caching files from GitHub releases and Hugging Face repositories.
///
/// # Supported Formats
///
/// | Source | Format | Example |
/// |--------|--------|---------|
/// | Local file | File path | `"./model.onnx"` |
/// | GitHub Release URL | Full URL | `"https://github.com/<owner>/<repo>/releases/download/<tag>/<file>"` |
/// | GitHub Release | `<tag>/<file>` | `"yolo/v5-n-det.onnx"` |
/// | HF URL | Full URL | `"https://huggingface.co/<owner>/<repo>/resolve/main/<file>"` |
/// | HF (inline) | `<owner>/<repo>/<file>` | `"BAAI/bge-m3/tokenizer.json"` |
/// | HF (dedicated) | `<file>` (via `from_hf`) | `"onnx/model.onnx"` |
///
/// # Examples
///
/// ```rust,ignore
/// // GitHub Release
/// let mut hub = Hub::default();
/// let path = hub.try_fetch("images/bus.jpg")?;
///
/// // HF (inline)
/// let path = Hub::default().try_fetch("<owner>/<repo>/<file>")?;
/// let path = Hub::default().try_fetch("<owner>/<repo>/<folder>/<file>")?;
///
/// // HF (dedicated)
/// let mut hub = Hub::from_hf("<owner>", "<repo>")?;
/// let path = hub.try_fetch("<file>")?;
///
/// // HF URL (resolve or blob)
/// let path = Hub::default().try_fetch(
///     "https://huggingface.co/<owner>/<repo>/blob/main/<file>"
/// )?;
/// ```
///
/// # HF Endpoint
/// Default: `https://huggingface.co`. Override via `HF_ENDPOINT` env var.
///
/// # Errors
/// Methods return `Result`. Errors may occur due to invalid paths, failed
/// network requests, cache write failures, or mismatched file sizes.
///
#[derive(Debug)]
pub struct Hub {
    /// GitHub repository owner
    owner: String,

    /// GitHub repository name
    repo: String,

    /// Directory to store the downloaded file
    #[allow(dead_code)]
    to: Dir,

    /// Time to live (cache duration)
    ttl: Duration,

    /// The maximum number of retry attempts for failed downloads or network operations
    max_attempts: u32,

    /// HF Endpoint
    hf_endpoint: String,

    /// Hugging Face repository (owner, repo). When set, try_fetch uses HF download path.
    hf_repo: Option<(String, String)>,
}

impl Default for Hub {
    fn default() -> Self {
        let owner = "jamjamjon".to_string();
        let repo = "assets".to_string();
        let max_attempts = 3;
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
            max_attempts,
            hf_endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".to_string()),
            hf_repo: None,
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

    pub fn from_hf(owner: &str, repo: &str) -> Result<Self> {
        Ok(Self {
            hf_repo: Some((owner.into(), repo.into())),
            ..Default::default()
        })
    }

    /// Fetches a file from local path, GitHub release, HF URL, or HF repository.
    ///
    /// Resolution order:
    /// 1. Local file path
    /// 2. Already cached locally (no network access needed)
    /// 3. GitHub release URL
    /// 4. Hugging Face URL (`/resolve/` or `/blob/`)
    /// 5. Dedicated HF repo (if `from_hf` / `with_hf_repo` was used)
    /// 6. Inline HF path (`<owner>/<repo>/<file>`, when path has 3+ segments)
    /// 7. Default GitHub release (`<tag>/<file>`)
    pub fn try_fetch(&mut self, s: &str) -> Result<String> {
        let span = tracing::info_span!("hub_fetch", source = s);
        let _enter = span.enter();

        // 1. Local file path
        let p = PathBuf::from(s);
        if p.exists() {
            tracing::debug!("Local file accessible: {}", p.display());
            return p
                .to_str()
                .map(|s| s.to_string())
                .with_context(|| format!("Failed to convert PathBuf: {p:?} to String"));
        }

        // 2. Check if already cached locally
        if let Some(cached) = self.resolve_cache_path(s)? {
            if cached.is_file() {
                tracing::debug!("Cache file available: {}", cached.display());
                return cached
                    .to_str()
                    .map(|s| s.to_string())
                    .with_context(|| format!("Failed to convert PathBuf: {cached:?} to String"));
            }
        }

        // 3. Not cached — resolve source and download
        let saveout = if let Some((owner_, repo_, tag_, file_name_)) =
            Self::is_valid_github_release_url(s)
        {
            // => GitHub release URL
            tracing::debug!(
                "Downloading from explicit GitHub URL: {}/{} (tag: {}, file: {})",
                owner_,
                repo_,
                tag_,
                file_name_
            );
            let saveout = self
                .to
                .crate_dir_default_with_subs(&[&owner_, &repo_, &tag_])?
                .join(&file_name_);
            retry!(
                self.max_attempts,
                1000,
                3000,
                self.download(s, &saveout, Some(&format!("{tag_}/{file_name_}")))
            )?;
            saveout
        } else if let Some((hf_owner, hf_repo_name, filename)) = Self::parse_hf_url(s) {
            // => Hugging Face URL (resolve/blob)
            tracing::debug!(
                "Downloading from HF URL: {}/{} -> {}",
                hf_owner,
                hf_repo_name,
                filename
            );
            return self.download_hf(&hf_owner, &hf_repo_name, &filename);
        } else if let Some((hf_owner, hf_repo)) = self.hf_repo.clone() {
            // => Dedicated HF mode
            tracing::debug!(
                "Downloading from dedicated HF repo: {}/{} -> {}",
                hf_owner,
                hf_repo,
                s
            );
            return self.download_hf(&hf_owner, &hf_repo, s);
        } else {
            let parts: Vec<&str> = s.split('/').filter(|x| !x.is_empty()).collect();
            if parts.len() > 2 {
                // => Inline HF path: owner/repo/filename
                let hf_owner = parts[0];
                let hf_repo_name = parts[1];
                let filename = parts[2..].join("/");
                tracing::debug!(
                    "Downloading from HF path: {}/{} -> {}",
                    hf_owner,
                    hf_repo_name,
                    filename
                );
                return self.download_hf(hf_owner, hf_repo_name, &filename);
            }

            // => Default hub (GitHub release tag/file format)
            tracing::debug!("Downloading from default GitHub release: {s}");
            match s.split_once('/') {
                Some((tag_, file_name_)) => {
                    let dst = self
                        .to
                        .crate_dir_default_with_subs(&[tag_])?
                        .join(file_name_);
                    let releases = self
                        .get_releases(&self.owner, &self.repo, &self.to, &self.ttl)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "No releases found in {}/{}. Error: {e}",
                                self.owner,
                                self.repo
                            )
                        })?;

                    let release =
                        releases
                            .iter()
                            .find(|r| r.tag_name == tag_)
                            .with_context(|| {
                                let tags: Vec<&str> =
                                    releases.iter().map(|r| r.tag_name.as_str()).collect();
                                format!("Tag `{tag_}` not found. Available: {tags:?}")
                            })?;

                    let asset = release
                        .assets
                        .iter()
                        .find(|a| a.name == file_name_)
                        .with_context(|| {
                            let files: Vec<&str> =
                                release.assets.iter().map(|a| a.name.as_str()).collect();
                            format!("File `{file_name_}` not in tag `{tag_}`. Available: {files:?}")
                        })?;

                    retry!(
                        self.max_attempts,
                        1000,
                        3000,
                        self.download(
                            &asset.browser_download_url,
                            &dst,
                            Some(&format!("{tag_}/{file_name_}")),
                        )
                    )?;
                    dst
                }
                _ => anyhow::bail!("Invalid format. Expected: <tag>/<file>, got: {s}"),
            }
        };

        tracing::debug!("Download completed: {}", saveout.display());
        saveout
            .to_str()
            .map(|s| s.to_string())
            .with_context(|| format!("Failed to convert PathBuf: {saveout:?} to String"))
    }

    /// Check if a file is already available locally (as a local file or in cache),
    /// without downloading. Returns `Some(path)` if found, `None` otherwise.
    pub fn cached(&self, s: &str) -> Option<String> {
        let p = PathBuf::from(s);
        if p.exists() {
            return p.to_str().map(|s| s.to_string());
        }
        if let Ok(Some(cached)) = self.resolve_cache_path(s) {
            if cached.is_file() {
                return cached.to_str().map(|s| s.to_string());
            }
        }
        None
    }

    /// Resolves the expected local cache path for a source string without any network calls.
    ///
    /// Returns `Ok(Some(path))` if a cache path can be determined, `Ok(None)` otherwise.
    fn resolve_cache_path(&self, s: &str) -> Result<Option<PathBuf>> {
        // GitHub release URL
        if let Some((owner_, repo_, tag_, file_name_)) = Self::is_valid_github_release_url(s) {
            let path = self
                .to
                .crate_dir_default_with_subs(&[&owner_, &repo_, &tag_])?
                .join(&file_name_);
            return Ok(Some(path));
        }

        // HF URL (resolve/blob)
        if let Some((hf_owner, hf_repo_name, filename)) = Self::parse_hf_url(s) {
            let path = self
                .to
                .crate_dir_default_with_subs(&[&hf_owner, &hf_repo_name])?
                .join(&filename);
            return Ok(Some(path));
        }

        // Dedicated HF mode
        if let Some((ref hf_owner, ref hf_repo)) = self.hf_repo {
            let path = self
                .to
                .crate_dir_default_with_subs(&[hf_owner, hf_repo])?
                .join(s);
            return Ok(Some(path));
        }

        let parts: Vec<&str> = s.split('/').filter(|x| !x.is_empty()).collect();
        if parts.len() > 2 {
            // Inline HF path: owner/repo/filename
            let hf_owner = parts[0];
            let hf_repo_name = parts[1];
            let filename = parts[2..].join("/");
            let path = self
                .to
                .crate_dir_default_with_subs(&[hf_owner, hf_repo_name])?
                .join(&filename);
            return Ok(Some(path));
        }

        // Default GitHub release (tag/file)
        if let Some((tag_, file_name_)) = s.split_once('/') {
            let path = self
                .to
                .crate_dir_default_with_subs(&[tag_])?
                .join(file_name_);
            return Ok(Some(path));
        }

        Ok(None)
    }

    /// Download a file from Hugging Face, returning the cached local path.
    fn download_hf(&self, owner: &str, repo: &str, filename: &str) -> Result<String> {
        let saveout = self
            .to
            .crate_dir_default_with_subs(&[owner, repo])?
            .join(filename);
        let url = format!(
            "{}/{}/{}/resolve/main/{}",
            self.hf_endpoint, owner, repo, filename
        );
        retry!(
            self.max_attempts,
            1000,
            3000,
            self.download(&url, &saveout, Some(&format!("{owner}/{repo}/{filename}")),)
        )?;
        saveout
            .to_str()
            .map(|s| s.to_string())
            .with_context(|| format!("Failed to convert PathBuf: {saveout:?} to String"))
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
            .with_context(|| format!("Failed to create cache directory: {parent_dir:?}"))?;

        // Create temporary file
        let mut temp_file = tempfile::NamedTempFile::new_in(parent_dir)
            .context("Failed to create temporary cache file")?;

        // Write data to temporary file
        temp_file
            .write_all(body.as_bytes())
            .context("Failed to write to temporary cache file")?;

        // Persist temporary file as the cache
        temp_file
            .persist(cache_path)
            .with_context(|| format!("Failed to persist temporary cache file to {cache_path:?}"))?;

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
            tracing::debug!("No cache found, fetching data from GitHub");
            true
        } else {
            match std::fs::metadata(file)?.modified() {
                Err(_) => {
                    tracing::debug!("Cannot get file modified time, fetching new data from GitHub");
                    true
                }
                Ok(modified_time) => {
                    if std::time::SystemTime::now().duration_since(modified_time)? < *ttl {
                        tracing::debug!("Using cached data");
                        false
                    } else {
                        tracing::debug!("Cache expired, fetching new data from GitHub");
                        true
                    }
                }
            }
        };
        Ok(y)
    }

    /// Download a file from a URL to a local path with a progress bar.
    pub fn download<P: AsRef<Path> + std::fmt::Debug>(
        &self,
        src: &str,
        dst: P,
        message: Option<&str>,
    ) -> Result<()> {
        let span = tracing::info_span!("hub_download", url = src);
        let _enter = span.enter();

        let dst_path = dst.as_ref();

        // Ensure parent directory exists for the final destination
        if let Some(parent_dir) = dst_path.parent() {
            std::fs::create_dir_all(parent_dir)
                .with_context(|| format!("Failed to create parent directory: {parent_dir:?}"))?;
        }

        let resp = self.fetch_get_response(src)?;
        let ntotal = resp
            .headers()
            .get(ureq::http::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()?.parse::<u64>().ok());

        let mut pb = match ntotal {
            Some(n) => crate::PB::fetch(n),
            None => crate::PB::fetch_stream(),
        };
        if let Some(msg) = message {
            pb = pb.with_message(msg);
        }

        // Create temporary file in system temp directory (more secure and reliable)
        let mut temp_file = NamedTempFile::new()
            .context("Failed to create temporary download file in system temp directory")?;

        let mut reader = resp.into_body().into_reader();
        const BUFFER_SIZE: usize = 64 * 1024;
        let mut buffer = [0; BUFFER_SIZE];
        let mut downloaded_bytes = 0usize;

        // Download to temporary file
        loop {
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            temp_file
                .write_all(&buffer[..bytes_read])
                .context("Failed to write to temporary file")?;
            downloaded_bytes += bytes_read;
            pb.inc(bytes_read as u64);
        }

        // Verify download completeness (only when Content-Length is known)
        if let Some(ntotal) = ntotal {
            if downloaded_bytes as u64 != ntotal {
                anyhow::bail!(
                    "The downloaded file is incomplete. Expected: {ntotal} bytes, got: {downloaded_bytes} bytes"
                );
            }
        }

        // Only persist the temporary file to the final destination if download is complete
        temp_file.persist(dst_path).with_context(|| {
            format!("Failed to move temporary file to final destination: {dst_path:?}")
        })?;

        // Update the progress bar
        pb.finish(None);

        tracing::debug!("Successfully downloaded and verified file: {:?}", dst_path);
        Ok(())
    }

    fn fetch_get_response(&self, url: &str) -> anyhow::Result<ureq::http::Response<ureq::Body>> {
        let config = ureq::Agent::config_builder()
            .proxy(ureq::Proxy::try_from_env())
            .build();
        let agent = ureq::Agent::new_with_config(config);

        // For development, Check for GITHUB_TOKEN environment variable to increase API rate limit
        // Without token: 60 requests/hour, with token: 5000 requests/hour
        let mut request = agent.get(url);
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            request = request.header("Authorization", &format!("Bearer {token}"));
        }

        let response = request
            .call()
            .map_err(|err| anyhow::anyhow!("Failed to GET response from {url}: {err}"))?;
        if response.status() != 200 {
            anyhow::bail!("Failed to fetch data from remote due to: {response:?}");
        }

        Ok(response)
    }

    #[allow(dead_code)]
    fn cache_file(owner: &str, repo: &str) -> String {
        let safe_owner = owner.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let safe_repo = repo.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        format!("releases-{safe_owner}-{safe_repo}.json")
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
            let gh_api_release =
                format!("https://api.github.com/repos/{owner}/{repo}/releases?per_page=100");
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

    /// Parse a Hugging Face URL into (owner, repo, filename).
    ///
    /// Supports both `/resolve/` and `/blob/` URL formats:
    /// - `https://huggingface.co/<owner>/<repo>/resolve/main/<file>?download=true`
    /// - `https://hf-mirror.com/<owner>/<repo>/blob/main/<file>`
    pub fn parse_hf_url(url: &str) -> Option<(String, String, String)> {
        let re =
            Regex::new(r"^https?://[^/]+/([^/]+)/([^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$")
                .ok()?;
        let caps = re.captures(url)?;
        Some((
            caps.get(1)?.as_str().to_string(),
            caps.get(2)?.as_str().to_string(),
            caps.get(3)?.as_str().to_string(),
        ))
    }

    pub fn with_owner(mut self, owner: &str) -> Self {
        self.owner = owner.to_string();
        self
    }

    pub fn with_repo(mut self, repo: &str) -> Self {
        self.repo = repo.to_string();
        self
    }

    pub fn with_hf_repo(mut self, owner: &str, repo: &str) -> Self {
        self.hf_repo = Some((owner.into(), repo.into()));
        self
    }

    pub fn with_hf_endpoint(mut self, x: &str) -> Self {
        self.hf_endpoint = x.to_string();
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

    /// Displays repository information including files and releases.
    ///
    /// For Hugging Face repositories, shows commit SHA and file tree.
    /// For GitHub repositories, shows releases with file counts and provides usage examples.
    pub fn info(&self) -> Result<()> {
        if let Some((hf_owner, hf_repo)) = &self.hf_repo {
            println!("Repository: {hf_owner}/{hf_repo}");
            println!("Type: Hugging Face Repository");
            println!("Endpoint: {}", self.hf_endpoint);
            let url = format!(
                "{}/api/models/{}/{}?blobs=true",
                self.hf_endpoint, hf_owner, hf_repo
            );
            match self.fetch_get_response(&url) {
                Ok(resp) => {
                    let body: serde_json::Value = serde_json::from_str(
                        &resp
                            .into_body()
                            .read_to_string()
                            .context("Failed to read HF API response")?,
                    )?;
                    if let Some(sha) = body["sha"].as_str() {
                        println!("Commit SHA: {sha}");
                    }
                    if let Some(siblings) = body["siblings"].as_array() {
                        println!("\nFiles ({} total):", siblings.len());
                        let mut files: Vec<(String, Option<u64>)> = siblings
                            .iter()
                            .filter_map(|s| {
                                let name = s["rfilename"].as_str()?.to_string();
                                let size = s["size"].as_u64().or_else(|| {
                                    s["lfs"].as_object().and_then(|lfs| lfs["size"].as_u64())
                                });
                                Some((name, size))
                            })
                            .collect();
                        files.sort_by(|a, b| a.0.cmp(&b.0));
                        let refs: Vec<(&str, Option<u64>)> =
                            files.iter().map(|(n, s)| (n.as_str(), *s)).collect();
                        Self::print_tree(&refs, "");
                    }
                }
                Err(e) => {
                    println!("Failed to fetch repository info: {e}");
                    println!("Visit: {}/{}/{}", self.hf_endpoint, hf_owner, hf_repo);
                }
            }
            return Ok(());
        }

        println!("Repository: {}/{}", self.owner, self.repo);
        println!("Type: GitHub Release Repository");
        let releases = self
            .get_releases(&self.owner, &self.repo, &self.to, &self.ttl)
            .unwrap_or_default();

        if releases.is_empty() {
            println!("No releases found in this repository.");
        } else {
            println!("\nReleases ({} total):", releases.len());
            for release in &releases {
                println!("  {} ({} files):", release.tag_name, release.assets.len());

                if release.assets.is_empty() {
                    println!("    (no files)");
                } else {
                    let files: Vec<(&str, Option<u64>)> = release
                        .assets
                        .iter()
                        .map(|a| (a.name.as_str(), Some(a.size)))
                        .collect();
                    if files.len() <= 5 {
                        Self::print_tree(&files, "    ");
                    } else {
                        Self::print_tree(&files[..5], "    ");
                        println!("    ... and {} more files", files.len() - 5);
                    }
                }
                println!();
            }

            println!("\nTip: Use the following code to get complete file list for all tags:");
            println!("\n```rust");
            println!("let hub = Hub::default().with_owner(\"jamjamjon\").with_repo(\"assets\");");
            println!("for tag in hub.tags().iter() {{");
            println!("    let files = hub.files(tag);");
            println!("    println!(\"Tag: {{}}, Files: {{:?}}\", tag, files);");
            println!("}}");
            println!("```\n");
        }

        Ok(())
    }

    fn print_tree(files: &[(&str, Option<u64>)], prefix: &str) {
        use std::collections::HashMap;

        #[derive(Default)]
        struct TreeNode {
            children: HashMap<String, TreeNode>,
            is_file: bool,
            size: Option<u64>,
        }

        let mut root = TreeNode::default();

        // Build tree structure
        for (file_path, size) in files {
            let parts: Vec<&str> = file_path.split('/').collect();
            let mut current = &mut root;

            for (i, part) in parts.iter().enumerate() {
                let is_last = i == parts.len() - 1;
                current = current.children.entry(part.to_string()).or_default();
                if is_last {
                    current.is_file = true;
                    current.size = *size;
                }
            }
        }

        // Print tree
        fn print_node(node: &TreeNode, name: &str, prefix: &str, is_last: bool, is_root: bool) {
            if !is_root {
                let connector = if is_last { "└── " } else { "├── " };
                if node.is_file {
                    if let Some(size) = node.size {
                        println!(
                            "{prefix}{connector}{name}  [{}]",
                            crate::human_bytes_binary(size as f64, 3)
                        );
                    } else {
                        println!("{prefix}{connector}{name}");
                    }
                } else {
                    println!("{prefix}{connector}{name}");
                }
            }

            let mut children: Vec<_> = node.children.iter().collect();
            children.sort_by_key(|(name, child)| (!child.is_file, name.as_str()));

            for (i, (child_name, child_node)) in children.iter().enumerate() {
                let is_last_child = i == children.len() - 1;
                let new_prefix = if is_root {
                    prefix.to_string()
                } else {
                    format!("{}{}   ", prefix, if is_last { " " } else { "│" })
                };
                print_node(child_node, child_name, &new_prefix, is_last_child, false);
            }
        }

        print_node(&root, "", prefix, true, true);
    }
}
