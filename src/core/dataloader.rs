use anyhow::{anyhow, Result};
use image::DynamicImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use video_rs::{Decoder, Url};
use walkdir::{DirEntry, WalkDir}; // TODO: remove

use crate::{Dir, Hub, ImageType, MediaType, VideoType, CHECK_MARK};

/// Dataloader for loading images, videos and streams,
// #[derive(Debug, Clone)]
pub struct DataLoader {
    pub paths: VecDeque<PathBuf>,
    pub media_types: VecDeque<MediaType>,
    pub recursive: bool, // TODO: remove
    pub batch: usize,
    pub decoder: Option<video_rs::decode::Decoder>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            batch: 1,
            recursive: false,
            paths: Default::default(),
            media_types: Default::default(),
            decoder: None,
        }
    }
}

// TODO: asycn or multi-threads
impl Iterator for DataLoader {
    type Item = (Vec<DynamicImage>, Vec<PathBuf>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut yis: Vec<DynamicImage> = Vec::new();
        let mut yps: Vec<PathBuf> = Vec::new();

        while !self.paths.is_empty() {
            match self.media_types.front().unwrap() {
                MediaType::Image(ImageType::Local) => {
                    let path = self.paths.pop_front().unwrap();
                    let _media_type = self.media_types.pop_front().unwrap();
                    match Self::try_read(&path) {
                        Err(err) => {
                            println!("Error reading image from path {:?}: {:?}", path, err);
                        }
                        Ok(x) => {
                            yis.push(x);
                            yps.push(path);
                        }
                    }
                }
                MediaType::Image(ImageType::Remote) => {
                    let path = self.paths.pop_front().unwrap();
                    let _media_type = self.media_types.pop_front().unwrap();
                    let file_name = path.file_name().unwrap();
                    let p_tmp = Dir::Cache.path(Some("tmp")).ok()?.join(file_name);
                    Hub::download(path.to_str().unwrap(), &p_tmp, None, None, None).ok()?;
                    match Self::try_read(&p_tmp) {
                        Err(err) => {
                            println!(
                                "Error reading downloaded image from path {:?}: {:?}",
                                p_tmp, err
                            );
                        }
                        Ok(x) => {
                            yis.push(x);
                            yps.push(path);
                        }
                    }
                }
                MediaType::Video(ty) => {
                    let path = self.paths.front().unwrap();
                    if self.decoder.is_none() {
                        let location: video_rs::location::Location = match ty {
                            VideoType::Local => path.clone().into(),
                            _ => path.to_str().unwrap().parse::<Url>().unwrap().into(),
                        };
                        self.decoder = Some(Decoder::new(location).unwrap());
                    }

                    let decoder = self.decoder.as_mut().unwrap();
                    let (w, h) = decoder.size();
                    let mut frames = decoder.decode_iter();

                    // Decode up to batch size frames
                    for _ in 0..self.batch {
                        match frames.next() {
                            Some(Ok((ts, frame))) => {
                                let rgb8: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                                    match image::ImageBuffer::from_raw(
                                        w as _,
                                        h as _,
                                        frame.into_raw_vec_and_offset().0,
                                    ) {
                                        Some(x) => x,
                                        None => continue,
                                    };
                                let x = image::DynamicImage::from(rgb8);
                                yis.push(x);
                                yps.push(ts.to_string().into());
                            }
                            Some(Err(_)) => {
                                let _ = self.paths.pop_front().unwrap();
                                let _ = self.media_types.pop_front().unwrap();
                                break;
                            }
                            None => break,
                        }
                    }
                }

                _ => todo!(),
            }

            if yis.len() == self.batch || self.paths.is_empty() {
                break;
            }
        }

        if yis.is_empty() {
            None
        } else {
            Some((yis, yps))
        }
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        let mut paths = VecDeque::new();
        let mut media_types = VecDeque::new();

        // local or remote
        let source_path = Path::new(source);
        if source_path.exists() {
            if source_path.is_file() {
                let file_type = MediaType::from_path(source_path);
                paths.push_back(source_path.to_path_buf());
                media_types.push_back(file_type);
            } else if source_path.is_dir() {
                for entry in (source_path.read_dir().map_err(|e| e.to_string()).unwrap()).flatten()
                {
                    if entry.path().is_file() {
                        let file_type = MediaType::from_path(&entry.path());

                        paths.push_back(entry.path());
                        media_types.push_back(file_type);
                    }
                }
            }
        } else {
            let media_type = MediaType::from_url(source);
            paths.push_back(PathBuf::from(source));
            media_types.push_back(media_type);
        }

        println!(
            "{CHECK_MARK} Media Type: {:?} x{}",
            media_types[0],
            paths.len()
        );
        Ok(DataLoader {
            paths,
            media_types,
            ..Default::default()
        })
    }

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
            s if !s.exists() => {
                // try download
                let p = Hub::new()?.fetch(s.to_str().unwrap())?.commit()?;
                let p = PathBuf::from(&p);
                VecDeque::from([p.to_path_buf()])
            }
            _ => todo!(),
        };
        println!("{CHECK_MARK} Found file x{}", self.paths.len());
        Ok(self)
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let mut path = path.as_ref().to_path_buf();

        // try to download
        if !path.exists() {
            let p = Hub::new()?.fetch(path.to_str().unwrap())?.commit()?;
            path = PathBuf::from(&p);
        }
        let img = image::ImageReader::open(&path)
            .map_err(|err| {
                anyhow!(
                    "Failed to open image at {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?
            .with_guessed_format()
            .map_err(|err| {
                anyhow!(
                    "Failed to make a format guess based on the content: {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?
            .decode()
            .map_err(|err| {
                anyhow!(
                    "Failed to decode image at {:?}. Error: {:?}",
                    path.display(),
                    err
                )
            })?
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
