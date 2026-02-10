use aksr::Builder;
use anyhow::Result;
use std::collections::HashSet;

use crate::{try_fetch_file_stem, DType, Device, EpConfig, Hub, Iiix};

/// ONNX Runtime configuration with device and optimization settings.
#[derive(Builder, Debug, Clone)]
pub struct ORTConfig {
    pub file: String,
    pub external_data_file: bool,
    pub device: Device,
    pub iiixs: Vec<Iiix>,
    pub num_dry_run: usize,
    pub spec: String, // TODO
    pub dtype: DType, // For dynamically loading the model
    pub graph_opt_level: Option<u8>,
    pub num_intra_threads: Option<usize>,
    pub num_inter_threads: Option<usize>,
    pub ep: EpConfig,
}

impl Default for ORTConfig {
    fn default() -> Self {
        Self {
            file: Default::default(),
            external_data_file: false,
            device: Default::default(),
            iiixs: Default::default(),
            spec: Default::default(),
            dtype: Default::default(),
            num_dry_run: 3,
            graph_opt_level: Default::default(),
            num_intra_threads: None,
            num_inter_threads: None,
            ep: EpConfig::default(),
        }
    }
}

impl ORTConfig {
    pub fn try_commit(mut self, name: &str) -> Result<Self> {
        tracing::debug!(
            "Model commit: resolving '{}' with file '{}'",
            name,
            self.file
        );

        // Identify the local model or fetch the remote model
        if std::path::PathBuf::from(&self.file).exists() {
            // Local file detected - no download required
            tracing::debug!("Local model file found, skipping download: {}", &self.file);
            self.spec = format!("{}/{}", name, try_fetch_file_stem(&self.file)?);
        } else {
            if self.file.is_empty() && name.is_empty() {
                anyhow::bail!(
                    "Failed to commit model. Invalid model config: neither `name` nor `file` were specified. Failed to fetch model from HuggingFace Hub or GitHub release."
                )
            }

            // Remote
            match Hub::is_valid_github_release_url(&self.file) {
                Some((owner, repo, tag, _file_name)) => {
                    // Explicit GitHub release URL detected
                    tracing::debug!(
                        "Explicit GitHub URL detected: {}/{} (tag: {})",
                        owner,
                        repo,
                        tag
                    );
                    let stem = try_fetch_file_stem(&self.file)?;
                    self.spec = format!("{name}/{owner}-{repo}-{tag}-{stem}");
                    self.file = Hub::default().try_fetch(&self.file)?;
                }
                None => {
                    // Not an explicit GitHub URL — could be a HuggingFace Hub path or
                    // a GitHub release file.
                    //
                    // Determine whether to prepend `name` as a tag prefix:
                    // - 1 segment  (e.g. "model.onnx")
                    //     → bare filename, prepend `name` to form "tag/file"
                    // - 2+ segments (e.g. "tag/model.onnx" or "owner/repo/dir/model.onnx")
                    //     → already contains path structure, keep as-is
                    //
                    // Note: the actual HF-vs-GitHub distinction is handled inside
                    // `Hub::try_fetch` (≥3 segments → HF Hub, 2 segments → GitHub release).
                    let parts = self.file.split('/').filter(|s| !s.is_empty()).count();
                    if parts > 1 {
                        tracing::debug!(
                            "File path has directory structure, using as-is: {}",
                            self.file
                        );
                    } else {
                        tracing::debug!(
                            "Bare filename, prepending model name as tag: {}/{}",
                            name,
                            self.file
                        );
                        self.file = format!("{}/{}", name, self.file);
                    }

                    // Save original path for resolving external data files later
                    let ext_file = self.file.clone();

                    // Build candidate file paths based on DType:
                    // - Auto/Fp32: use the file path as-is (or "{dtype}.onnx" if empty)
                    // - Other dtypes + non-empty file: try "model{delim}{dtype}.onnx"
                    //   with delimiters ['-', '_', '.'] (e.g., "model-fp16.onnx")
                    // - Other dtypes + empty file: use "{dtype}.onnx" directly
                    let candidates: Vec<String> = match self.dtype {
                        d @ (DType::Auto | DType::Fp32) => {
                            if self.file.is_empty() {
                                vec![format!("{d}.onnx")]
                            } else {
                                vec![self.file.clone()]
                            }
                        }
                        dtype => {
                            if self.file.is_empty() {
                                vec![format!("{dtype}.onnx")]
                            } else {
                                ['-', '_', '.']
                                    .iter()
                                    .map(|delim| {
                                        let mut base = self.file.clone();
                                        let suffix = base.split_off(base.len() - 5); // 5 -> ".onnx"
                                        format!("{base}{delim}{dtype}{suffix}")
                                    })
                                    .collect()
                            }
                        }
                    };
                    tracing::debug!(
                        "Generated {} candidate file paths for resolution: {:?}",
                        candidates.len(),
                        candidates
                    );

                    // Phase 1: Check if any candidate is already cached locally (no HTTP)
                    let mut hub = Hub::default();
                    let mut fetch_success = false;
                    for file in &candidates {
                        if let Some(cached_path) = hub.cached(file) {
                            self.file = cached_path;
                            let stem = try_fetch_file_stem(file)?;
                            self.spec = format!("{name}/{stem}");
                            tracing::debug!("Cache hit: {} -> {}", file, &self.file);
                            fetch_success = true;
                            break;
                        }
                    }

                    // Phase 2: Nothing cached — try fetching each candidate from remote
                    if !fetch_success {
                        for file in &candidates {
                            tracing::debug!("Requesting Hub to download: {file}");
                            match hub.try_fetch(file) {
                                Ok(f) => {
                                    self.file = f;
                                    let stem = try_fetch_file_stem(file)?;
                                    self.spec = format!("{name}/{stem}");
                                    tracing::debug!(
                                        "Successfully resolved candidate '{}' to spec: {}",
                                        file,
                                        &self.spec
                                    );
                                    fetch_success = true;
                                    break;
                                }
                                Err(err) => {
                                    tracing::warn!("Failed to download candidate '{file}': {err}");
                                }
                            }
                        }
                    }

                    if !fetch_success {
                        anyhow::bail!(
                            "Failed to fetch ONNX model file. \
                             Neither a GitHub release file nor a HuggingFace Hub file \
                             could be resolved. \
                             Please verify the model file path: {:?}",
                            self.file
                        );
                    }

                    // Attempt to fetch external data files referenced by the model
                    let proto = crate::load_onnx(&self.file)?;
                    let graph = proto.graph.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "No graph found in ONNX proto. Invalid model: {}",
                            self.file
                        )
                    })?;

                    // Collect all external data file locations
                    let external_files: HashSet<String> = graph
                        .initializer
                        .iter()
                        .filter(|t| t.data_location == 1)
                        .flat_map(|t| t.external_data.iter())
                        .filter(|x| x.key == "location")
                        .map(|x| x.value.clone())
                        .collect();

                    if !external_files.is_empty() {
                        tracing::debug!(
                            "Found {} external data files, requesting downloads: {:?}",
                            external_files.len(),
                            external_files
                        );
                    }

                    // Download all external data files
                    let base_path = ext_file.rsplit_once('/').map(|(base, _)| base);
                    for f in &external_files {
                        let base = base_path.ok_or_else(|| {
                            anyhow::anyhow!(
                                "Cannot resolve external data file path: \
                                 no parent directory in '{ext_file}'"
                            )
                        })?;
                        let file_path = format!("{base}/{f}");
                        match Hub::default().try_fetch(&file_path) {
                            Ok(local) => {
                                tracing::debug!(
                                    "Successfully fetched external data file: {} -> {}",
                                    file_path,
                                    local
                                );
                            }
                            Err(err) => {
                                anyhow::bail!(
                                    "Found external data reference '{file_path}' \
                                     but failed to fetch it: {err}"
                                );
                            }
                        }
                    }
                }
            }
        }

        tracing::debug!(
            "Model commit completed: spec='{}', file='{}'",
            self.spec,
            self.file
        );
        Ok(self)
    }
}
