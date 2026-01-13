use anyhow::Result;
use image::DynamicImage;
use std::path::{Path, PathBuf};

use crate::load_paths;

pub const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "tif"];
pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "mpeg", "mpg", "m4v", "m4p",
];
pub const REMOTE_PROTOCOLS: &[&str] = &[
    "http://", "https://", "ftp://", "ftps://", "sftp://", "mms://", "mmsh://", "rtsp://",
    "rtmp://", "rtmps://", "file://",
];
pub const STREAM_PROTOCOLS: &[&str] = &[
    "rtsp://", "rtsps://", "rtspu://", "rtmp://", "rtmps://", "hls://",
];

/// Represents an atomic unit of input for inference.
#[derive(Debug, Clone)]
pub enum SourceType {
    Image(PathBuf),
    ImageUrl(String),
    DynamicImage(DynamicImage),
    Video(PathBuf),
    Webcam(u32),
    Stream(String),
    Directory(PathBuf),
    Glob(String),
}

impl From<PathBuf> for SourceType {
    fn from(p: PathBuf) -> Self {
        p.to_string_lossy()
            .as_ref()
            .parse()
            .unwrap_or(Self::Image(p))
    }
}

impl From<&PathBuf> for SourceType {
    fn from(p: &PathBuf) -> Self {
        p.to_string_lossy()
            .as_ref()
            .parse()
            .unwrap_or_else(|_| Self::Image(p.clone()))
    }
}

impl From<&Path> for SourceType {
    fn from(p: &Path) -> Self {
        p.to_string_lossy()
            .as_ref()
            .parse()
            .unwrap_or_else(|_| Self::Image(p.to_path_buf()))
    }
}

impl From<String> for SourceType {
    fn from(s: String) -> Self {
        s.parse().unwrap_or_else(|_| Self::Image(PathBuf::from(s)))
    }
}

impl From<&str> for SourceType {
    fn from(s: &str) -> Self {
        s.parse().unwrap_or_else(|_| Self::Image(PathBuf::from(s)))
    }
}

impl From<&String> for SourceType {
    fn from(s: &String) -> Self {
        s.as_str()
            .parse()
            .unwrap_or_else(|_| Self::Image(PathBuf::from(s)))
    }
}

impl From<DynamicImage> for SourceType {
    fn from(img: DynamicImage) -> Self {
        Self::DynamicImage(img)
    }
}

impl From<u32> for SourceType {
    fn from(idx: u32) -> Self {
        Self::Webcam(idx)
    }
}

impl From<i32> for SourceType {
    fn from(idx: i32) -> Self {
        Self::Webcam(idx as u32)
    }
}

impl std::str::FromStr for SourceType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            anyhow::bail!("Empty source string");
        }

        // 1. Check if it's a remote URL or Stream
        if REMOTE_PROTOCOLS.iter().any(|&p| s.starts_with(p)) {
            if STREAM_PROTOCOLS.iter().any(|&p| s.starts_with(p)) {
                return Ok(Self::Stream(s.to_string()));
            }

            let url_lower = s.to_lowercase();
            let path_part = url_lower.split('?').next().unwrap_or(&url_lower);
            let ext = Path::new(path_part)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                return Ok(Self::ImageUrl(s.to_string()));
            } else if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
                return Ok(Self::Video(PathBuf::from(s)));
            }

            return Ok(Self::Stream(s.to_string()));
        }

        // 2. Check if it's a webcam index
        if let Ok(index) = s.parse::<u32>() {
            return Ok(Self::Webcam(index));
        }

        // 3. Check if it's a glob pattern
        if (s.contains('*') || s.contains('?') || (s.contains('[') && s.contains(']')))
            && glob::Pattern::new(s).is_ok()
        {
            return Ok(Self::Glob(s.to_string()));
        }

        // 4. Check local filesystem or identify by extension
        let path = Path::new(s);
        if path.exists() {
            if path.is_dir() {
                return Ok(Self::Directory(path.to_path_buf()));
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                return Ok(Self::Image(path.to_path_buf()));
            } else if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
                return Ok(Self::Video(path.to_path_buf()));
            }
        }

        // 5. Check Hub pattern (tag/filename.ext)
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() == 2 && !s.starts_with('.') && !s.starts_with('/') {
            let tag = parts[0];
            let filename = parts[1];
            let ext = Path::new(filename)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            if IMAGE_EXTENSIONS.contains(&ext.as_str()) || VIDEO_EXTENSIONS.contains(&ext.as_str())
            {
                // Only instantiate Hub if the pattern matches, to save resources
                let hub = crate::Hub::default();
                if hub.tags().contains(&tag.to_string()) {
                    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                        return Ok(Self::ImageUrl(s.to_string()));
                    } else {
                        return Ok(Self::Video(PathBuf::from(s)));
                    }
                }
            }
        }

        // Final fallback: identify by extension even if path doesn't exist yet
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            return Ok(Self::Image(path.to_path_buf()));
        } else if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
            return Ok(Self::Video(path.to_path_buf()));
        }

        anyhow::bail!("Could not identify source type: {s}")
    }
}

impl SourceType {
    pub const fn is_image(&self) -> bool {
        matches!(
            self,
            Self::Image(_) | Self::DynamicImage(_) | Self::ImageUrl(_)
        )
    }

    pub const fn is_video(&self) -> bool {
        matches!(self, Self::Video(_) | Self::Webcam(_) | Self::Stream(_))
    }

    pub fn flatten(self) -> Result<Vec<SourceType>> {
        match self {
            Self::Directory(path) => {
                let p_str = path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid path: {path:?}"))?;

                let mut all_exts = IMAGE_EXTENSIONS.to_vec();
                all_exts.extend_from_slice(VIDEO_EXTENSIONS);
                let found = load_paths(p_str, Some(&all_exts), true)?;

                let results: Vec<SourceType> = found
                    .into_iter()
                    .filter_map(|p| {
                        let ext = p.extension()?.to_str()?.to_lowercase();
                        if IMAGE_EXTENSIONS.iter().any(|&e| e == ext) {
                            Some(SourceType::Image(p))
                        } else if VIDEO_EXTENSIONS.iter().any(|&e| e == ext) {
                            Some(SourceType::Video(p))
                        } else {
                            None
                        }
                    })
                    .collect();

                Ok(results)
            }
            Self::Glob(pattern) => {
                let mut all_exts = IMAGE_EXTENSIONS.to_vec();
                all_exts.extend_from_slice(VIDEO_EXTENSIONS);
                let found = load_paths(&pattern, Some(&all_exts), true)?;
                let results: Vec<SourceType> = found
                    .into_iter()
                    .filter_map(|p| {
                        let ext = p.extension()?.to_str()?.to_lowercase();
                        if IMAGE_EXTENSIONS.iter().any(|&e| e == ext) {
                            Some(SourceType::Image(p))
                        } else if VIDEO_EXTENSIONS.iter().any(|&e| e == ext) {
                            Some(SourceType::Video(p))
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(results)
            }
            _ => Ok(vec![self]),
        }
    }
}
