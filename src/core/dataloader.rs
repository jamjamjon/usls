use crate::{CHECK_MARK, SAFE_CROSS_MARK};
use anyhow::{anyhow, bail, Result};
use image::DynamicImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

/// Dataloader for load images
#[derive(Debug, Clone)]
pub struct DataLoader {
    pub paths: VecDeque<PathBuf>,
    pub recursive: bool,
    pub batch: usize,
}

impl Iterator for DataLoader {
    type Item = (Vec<DynamicImage>, Vec<PathBuf>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.paths.is_empty() {
            None
        } else {
            let mut yis: Vec<DynamicImage> = Vec::new();
            let mut yps: Vec<PathBuf> = Vec::new();
            loop {
                let path = self.paths.pop_front().unwrap();
                match Self::try_read(&path) {
                    Err(err) => {
                        println!("{SAFE_CROSS_MARK} {err}");
                    }
                    Ok(x) => {
                        yis.push(x);
                        yps.push(path);
                    }
                }
                if self.paths.is_empty() || yis.len() == self.batch {
                    break;
                }
            }
            Some((yis, yps))
        }
    }
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            batch: 1,
            recursive: false,
            paths: Default::default(),
        }
    }
}

impl DataLoader {
    pub fn load<P: AsRef<Path>>(mut self, source: P) -> Result<Self> {
        self.paths = match source.as_ref() {
            s if s.is_file() => VecDeque::from([s.to_path_buf()]),
            s if s.is_dir() => WalkDir::new(s)
                .into_iter()
                .filter_entry(|e| !Self::_is_hidden(e))
                .filter_map(|entry| match entry {
                    Err(_) => None,
                    Ok(entry) => {
                        if entry.file_type().is_dir() {
                            return None;
                        }
                        if !self.recursive && entry.depth() > 1 {
                            return None;
                        }
                        Some(entry.path().to_path_buf())
                    }
                })
                .collect::<VecDeque<_>>(),
            // s if s.starts_with("rtsp://") || s.starts_with("rtmp://") || s.starts_with("http://")|| s.starts_with("https://") => todo!(),
            s if !s.exists() => bail!("{s:?} Not Exists"),
            _ => todo!(),
        };
        println!("{CHECK_MARK} Found file x{}", self.paths.len());
        Ok(self)
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let img = image::ImageReader::open(&path)
            .map_err(|_| anyhow!("Failed to open image at {:?}", path.as_ref()))?
            .decode()
            .map_err(|_| anyhow!("Failed to decode image at {:?}", path.as_ref()))?
            .into_rgb8();
        Ok(DynamicImage::from(img))
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch = x;
        self
    }

    pub fn with_recursive(mut self, x: bool) -> Self {
        self.recursive = x;
        self
    }

    pub fn paths(&self) -> &VecDeque<PathBuf> {
        &self.paths
    }

    fn _is_hidden(entry: &DirEntry) -> bool {
        entry
            .file_name()
            .to_str()
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
    }
}
