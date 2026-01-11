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

/// Manages interactions with GitHub repository releases and Hugging Face repositories
///
/// # Format Rules
/// - **GitHub Release**
///   - Use `<tag>/<file>` format.
///   - Example: `"yolo/v5-n-det.onnx"`.
///
/// - **Hugging Face**
///   - With `Hub::from_hf(owner, repo)`: File paths are interpreted relative to the repo root.
///     - Example: `"sentencepiece.bpe.model"`, `"onnx/tokenizer.json"`.
///   - With `Hub::default()`: Paths with three segments are interpreted as `<owner>/<repo>/<file>`.
///     - Example: `"BAAI/bge-m3/sentencepiece.bpe.model"`.
///
/// # Examples
///
/// ## GitHub Release Download
/// ```rust,ignore
/// let mut hub = Hub::default();
/// // let mut hub = Hub::new(owner, repo);  // Optional: Specify owner and repo if not using default
/// let path = hub.try_fetch("images/bus.jpg")?; // <tag>/<file> format
/// ```
///
/// ## Hugging Face Download (Dedicated Hub)
/// ```rust,ignore
/// let mut hub = Hub::from_hf("BAAI", "bge-m3")?;
/// let path = hub.try_fetch("sentencepiece.bpe.model")?; // Any format works
/// let path = hub.try_fetch("onnx/tokenizer.json")?; // Any format works
/// ```
///
/// ## Hugging Face Download (Temporary)
/// ```rust,ignore
/// let mut hub = Hub::default().try_fetch("BAAI/bge-m3/tokenizer_config.json")?; // <owner>/<repo>/<file> format
/// ```
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
    #[allow(dead_code)]
    to: Dir,

    /// Time to live (cache duration)
    ttl: Duration,

    /// The maximum number of retry attempts for failed downloads or network operations
    max_attempts: u32,

    /// HF Endpoint (only used when hf-hub feature is enabled)
    #[cfg(feature = "hf-hub")]
    hf_endpoint: String,

    /// HF Api Repo (only used when hf-hub feature is enabled)
    #[cfg(feature = "hf-hub")]
    hf_repo: Option<hf_hub::api::sync::ApiRepo>,
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
            #[cfg(feature = "hf-hub")]
            hf_endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".to_string()),
            #[cfg(feature = "hf-hub")]
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

    #[cfg(feature = "hf-hub")]
    pub fn from_hf(owner: &str, repo: &str) -> Result<Self> {
        let mut self_ = Self::new(owner, repo);
        let hf_api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(
                self_
                    .to
                    .crate_dir_default()
                    .expect("Faild to get cache dir"),
            )
            .with_endpoint(self_.hf_endpoint.clone())
            .with_retries(self_.max_attempts as usize)
            .with_progress(true)
            .build()?;
        self_.hf_repo = Some(hf_api.model(format!("{}/{}", owner, repo)));

        Ok(self_)
    }

    #[cfg(not(feature = "hf-hub"))]
    pub fn from_hf(_owner: &str, _repo: &str) -> Result<Self> {
        anyhow::bail!("HF hub support is not enabled. Please enable the 'hf-hub' feature.")
    }

    /// Attempts to fetch a file from a local path, GitHub release, or Hugging Face repository.
    ///
    /// The `try_fetch` method supports multiple scenarios:
    /// 1. **Local file**: If the provided string is a valid file path, the file is returned without downloading.
    /// 2. **GitHub release URL**: If the input matches a valid GitHub release URL, the corresponding file is downloaded.
    /// 3. **Hugging Face repository**: If the hub is configured for HF or the path contains HF format, files are downloaded from HF.
    /// 4. **Default repository**: If no explicit URL is provided, the method uses the default or configured repository.
    ///
    /// # Parameters
    /// - `s`: A string representing the file to fetch. This can be:
    ///   - A local file path.
    ///   - A GitHub release URL (e.g., `https://github.com/owner/repo/releases/download/tag/file`).
    ///   - A `<tag>/<file>` format for fetching from the default GitHub repository.
    ///   - A HF repository file path (e.g., `"sentencepiece.bpe.model"` when using `from_hf`).
    ///   - A temporary HF path format (e.g., `"BAAI/bge-m3/sentencepiece.bpe.model"`).
    ///
    /// # Returns
    /// - `Result<String>`: On success, returns the path to the fetched file.
    ///
    /// # Errors
    /// - Returns an error if:
    ///   - The file cannot be found locally.
    ///   - The URL or tag is invalid.
    ///   - Network operations fail after the maximum retry attempts.
    ///   - HF repository access fails.
    ///
    /// # Examples
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
    /// // Fetch a file using the default GitHub repository
    /// let default_repo_path = hub.try_fetch("yolo/v5-n-det.onnx").expect("Failed to fetch file");
    ///
    /// // Method 1: Fetch from HF repository using dedicated hub
    /// let mut hf_hub = Hub::from_hf("BAAI", "bge-m3")?;
    /// let hf_path = hf_hub.try_fetch("sentencepiece.bpe.model").expect("Failed to fetch HF file");
    ///
    /// // Method 2: Fetch from HF repository using temporary path (doesn't change hub's owner/repo)
    /// let temp_hf_path = Hub::default().try_fetch("BAAI/bge-m3/sentencepiece.bpe.model")
    ///     .expect("Failed to fetch HF file");
    /// ```
    pub fn try_fetch(&mut self, s: &str) -> Result<String> {
        let span = tracing::info_span!("hub_fetch", source = s);
        let _enter = span.enter();

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
        } else {
            // First, check if it's a valid GitHub release URL
            // This must be checked BEFORE HF path check, because GitHub URLs also have parts.len() > 2
            if let Some((owner_, repo_, tag_, file_name_)) = Self::is_valid_github_release_url(s) {
                // => Valid GitHub release URL
                // keep original owner, repo and tag
                let saveout = self
                    .to
                    .crate_dir_default_with_subs(&[&owner_, &repo_, &tag_])?
                    .join(&file_name_);
                pack = pack.with_url(s).with_tag(&tag_).with_file_name(&file_name_);
                if let Some(n) = retry!(self.max_attempts, self.fetch_get_response(s))?
                    .headers()
                    .get(ureq::http::header::CONTENT_LENGTH)
                    .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                {
                    pack = pack.with_file_size(n);
                }

                saveout
            } else {
                // Check for HF repo usage (only after confirming it's not a GitHub URL)
                #[cfg(feature = "hf-hub")]
                {
                    if let Some(hf_repo) = &self.hf_repo {
                        return Ok(hf_repo.get(s)?.to_str().unwrap().to_string());
                        // from hf repo
                    }
                    let parts: Vec<&str> = s.split('/').filter(|x| !x.is_empty()).collect();
                    if parts.len() > 2 {
                        // from hf repo
                        // Note: this does not update self.owner or self.repo; they are only used temporarily.
                        let hf_api = hf_hub::api::sync::ApiBuilder::new()
                            .with_cache_dir(
                                self.to.crate_dir_default().expect("Faild to get cache dir"),
                            )
                            .with_endpoint(self.hf_endpoint.clone())
                            .with_progress(true)
                            .with_retries(self.max_attempts as usize)
                            .build()?;
                        let hf_repo = hf_api.model(format!("{}/{}", parts[0], parts[1]));
                        return Ok(hf_repo
                            .get(&parts[2..].join("/"))?
                            .to_str()
                            .unwrap()
                            .to_string());
                    }
                }
                #[cfg(not(feature = "hf-hub"))]
                {
                    let parts: Vec<&str> = s.split('/').filter(|x| !x.is_empty()).collect();
                    if parts.len() > 2 {
                        anyhow::bail!("HF hub support is not enabled. Please enable the 'hf-hub' feature to download from Hugging Face repositories.")
                    }
                }

                // => Default hub (GitHub release tag/file format)

                // Check remote
                match s.split_once('/') {
                            Some((tag_, file_name_)) => {
                                let dst = self.to
                                    .crate_dir_default_with_subs(&[tag_])?
                                    .join(file_name_);

                                // check if is cached
                                if !dst.is_file() {
                                    tracing::debug!("File not cached, fetching from remote: {}", dst.display());

                                    // Fetch releases
                                    let releases =
                                    match self.get_releases(&self.owner, &self.repo, &self.to, &self.ttl) {
                                        Err(err) => anyhow::bail!(
                                            "Failed to download: No releases found in this repo. Error: {}",
                                            err
                                        ),
                                        Ok(releases) => releases,
                                    };

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
                                    }
                                }
                                tracing::debug!("Using cached file: {}", dst.display());
                                dst
                            }
                            _ => anyhow::bail!(
                                "Failed to download file from github releases due to invalid format. Expected: <tag>/<file>, got: {}",
                                s
                            ),
                        }
            }
        };

        // Commit the downloaded file, downloading if necessary
        if !pack.url.is_empty() {
            tracing::debug!("Starting remote file download...");
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
            // // Download if the file does not exist or if the size of file does not match
            // if saveout.is_file() {
            //     match pack.file_size {
            //         None => {
            //             log::warn!(
            //                 "Failed to retrieve the remote file size. \
            //                 Download will be skipped, which may cause issues. \
            //                 Please verify your network connection or ensure the local file is valid and complete."
            //             );
            //         }
            //         Some(file_size) => {
            //             if std::fs::metadata(&saveout)?.len() != file_size {
            //                 tracing::debug!(
            //                     "Local file size does not match remote. Starting download."
            //                 );
            //                 retry!(
            //                     self.max_attempts,
            //                     1000,
            //                     3000,
            //                     self.download(
            //                         &pack.url,
            //                         &saveout,
            //                         Some(&format!("{}/{}", pack.tag, pack.file_name)),
            //                     )
            //                 )?;
            //             } else {
            //                 tracing::debug!("Local file size matches remote. No download required.");
            //             }
            //         }
            //     }
            // } else {
            //     tracing::debug!("Starting remote file download...");
            //     retry!(
            //         self.max_attempts,
            //         1000,
            //         3000,
            //         self.download(
            //             &pack.url,
            //             &saveout,
            //             Some(&format!("{}/{}", pack.tag, pack.file_name)),
            //         )
            //     )?;
            // }
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

    /// Download a file from a github release to a specified path with a progress bar
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
                .with_context(|| format!("Failed to create parent directory: {:?}", parent_dir))?;
        }

        let resp = self.fetch_get_response(src)?;
        let ntotal = resp
            .headers()
            .get(ureq::http::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
            .context("Content-Length header is missing or invalid")?;

        let mut pb = crate::PB::fetch(ntotal);
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

        // Verify download completeness
        if downloaded_bytes as u64 != ntotal {
            anyhow::bail!(
                "The downloaded file is incomplete. Expected: {} bytes, got: {} bytes",
                ntotal,
                downloaded_bytes
            );
        }

        // Only persist the temporary file to the final destination if download is complete
        temp_file.persist(dst_path).with_context(|| {
            format!(
                "Failed to move temporary file to final destination: {:?}",
                dst_path
            )
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
            request = request.header("Authorization", &format!("Bearer {}", token));
        }

        let response = request
            .call()
            .map_err(|err| anyhow::anyhow!("Failed to GET response from {}: {}", url, err))?;
        if response.status() != 200 {
            anyhow::bail!("Failed to fetch data from remote due to: {:?}", response);
        }

        Ok(response)
    }

    #[allow(dead_code)]
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

    #[cfg(feature = "hf-hub")]
    pub fn with_hf_owner_repo(self, owner: &str, repo: &str) -> Result<Self> {
        Self::from_hf(owner, repo)
    }

    #[cfg(feature = "hf-hub")]
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
        println!("Repository: {}/{}", self.owner, self.repo);

        #[cfg(feature = "hf-hub")]
        {
            if let Some(hf_repo) = &self.hf_repo {
                let info = hf_repo.info()?;

                println!("Type: Hugging Face Repository");
                println!("Commit SHA: {}", info.sha);

                println!("\nFiles ({} total):", info.siblings.len());
                let mut files: Vec<_> =
                    info.siblings.iter().map(|s| s.rfilename.as_str()).collect();
                files.sort();

                Self::print_tree(&files, "");
                return Ok(());
            }
        }

        println!("Type: GitHub Release Repository");
        let tags = self.tags();

        if tags.is_empty() {
            println!("No releases found in this repository.");
        } else {
            println!("\nReleases ({} total):", tags.len());
            for tag in &tags {
                let files = self.files(tag);
                println!("  {} ({} files):", tag, files.len());

                if files.is_empty() {
                    println!("    (no files)");
                } else if files.len() <= 5 {
                    // Show all files if 5 or fewer
                    let file_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
                    Self::print_tree(&file_refs, "    ");
                } else {
                    // Show first 5 files and indicate there are more
                    let file_refs: Vec<&str> = files.iter().take(5).map(|s| s.as_str()).collect();
                    Self::print_tree(&file_refs, "    ");
                    println!("    ... and {} more files", files.len() - 5);
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

    fn print_tree(files: &[&str], prefix: &str) {
        use std::collections::HashMap;

        #[derive(Default)]
        struct TreeNode {
            children: HashMap<String, TreeNode>,
            is_file: bool,
        }

        let mut root = TreeNode::default();

        // Build tree structure
        for file_path in files {
            let parts: Vec<&str> = file_path.split('/').collect();
            let mut current = &mut root;

            for (i, part) in parts.iter().enumerate() {
                let is_last = i == parts.len() - 1;
                current = current.children.entry(part.to_string()).or_default();
                if is_last {
                    current.is_file = true;
                }
            }
        }

        // Print tree
        fn print_node(node: &TreeNode, name: &str, prefix: &str, is_last: bool, is_root: bool) {
            if !is_root {
                let connector = if is_last { "└── " } else { "├── " };
                println!("{}{}{}", prefix, connector, name);
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
