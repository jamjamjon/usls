use crate::{CHECK_MARK, CROSS_MARK, SAFE_CROSS_MARK};
use anyhow::Result;
use image::DynamicImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use walkdir::{DirEntry, WalkDir};

#[derive(Debug, Clone)]
pub struct DataLoader {
    // source could be single image, folder with images (TODO: video, stream)
    pub source: PathBuf,
    pub batch: usize,
    pub recursive: bool,
    pub paths: VecDeque<PathBuf>,
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
                match image::io::Reader::open(&path) {
                    Err(err) => {
                        println!(
                            "{SAFE_CROSS_MARK} Faild to load image: {:?} -> {:?}",
                            self.paths[0], err
                        );
                    }
                    Ok(p) => match p.decode() {
                        Err(err) => {
                            println!(
                                "{SAFE_CROSS_MARK} Fail to load image: {:?} -> {:?}",
                                self.paths[0], err
                            );
                        }
                        Ok(x) => {
                            yis.push(x);
                            yps.push(path);
                        }
                    },
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
            source: Default::default(),
            paths: Default::default(),
        }
    }
}

impl DataLoader {
    pub fn load<P: AsRef<Path>>(&self, source: P) -> Result<Self> {
        let source = source.as_ref();
        let mut paths = VecDeque::new();

        match source {
            s if s.is_file() => paths.push_back(s.to_path_buf()),
            s if s.is_dir() => {
                for entry in WalkDir::new(s)
                    .into_iter()
                    .filter_entry(|e| !Self::_is_hidden(e))
                {
                    let entry = entry.unwrap();
                    if entry.file_type().is_dir() {
                        continue;
                    }
                    if !self.recursive && entry.depth() > 1 {
                        continue;
                    }
                    paths.push_back(entry.path().to_path_buf());
                }
            }
            // s if s.starts_with("rtsp://") || s.starts_with("rtmp://") || s.starts_with("http://")|| s.starts_with("https://") => todo!(),
            s if !s.exists() => panic!("{CROSS_MARK} File not found: {s:?}"),
            _ => todo!(),
        }
        println!("{CHECK_MARK} {} files found\n", &paths.len());
        Ok(Self {
            paths,
            source: source.into(),
            batch: self.batch,
            recursive: self.recursive,
        })
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch = x;
        self
    }

    pub fn with_recursive(mut self, x: bool) -> Self {
        self.recursive = x;
        self
    }

    fn _is_hidden(entry: &DirEntry) -> bool {
        entry
            .file_name()
            .to_str()
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
    }
}
