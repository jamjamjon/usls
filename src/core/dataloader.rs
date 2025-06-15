use anyhow::{anyhow, Result};
use glob::{glob_with, MatchOptions};
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, warn};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc;
#[cfg(feature = "video")]
use video_rs::{Decoder, Url};

use crate::{Image, Location, MediaType};

/// A structure designed to load and manage image, video, or stream data.
pub struct DataLoader {
    /// Queue of paths for images.
    paths: Option<VecDeque<PathBuf>>,

    /// Media type of the source (image, video, stream, etc.).
    media_type: MediaType,

    /// Batch size for iteration, determining how many files are processed at once.
    batch_size: usize,

    /// Buffer size for the channel, used to manage the buffer between producer and consumer.
    bound: Option<usize>,

    /// Receiver for processed data.
    receiver: mpsc::Receiver<Vec<Image>>,

    /// Video decoder for handling video or stream data.
    #[cfg(feature = "video")]
    decoder: Option<video_rs::decode::Decoder>,

    /// Number of images or frames; `u64::MAX` is used for live streams (indicating no limit).
    nf: u64,

    /// Number of frames to be skipped.
    #[cfg(feature = "video")]
    nf_skip: u64,

    /// Frame rate for video or stream.
    #[cfg(feature = "video")]
    frame_rate: f32,

    /// Progress bar for displaying iteration progress.
    progress_bar: Option<ProgressBar>,

    /// Display progress bar or not.
    with_progress_bar: bool,
}

impl Default for DataLoader {
    fn default() -> Self {
        DataLoader {
            paths: None,
            media_type: Default::default(),
            nf: 0,
            batch_size: 1,
            #[cfg(feature = "video")]
            nf_skip: 0,
            bound: None,
            receiver: mpsc::sync_channel(0).1,
            progress_bar: None,
            with_progress_bar: false,
            #[cfg(feature = "video")]
            decoder: None,
            #[cfg(feature = "video")]
            frame_rate: 25.0,
        }
    }
}

impl std::fmt::Debug for DataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoader")
            .field("paths", &self.paths)
            .field("batch_size", &self.batch_size)
            .field("nf", &self.nf)
            // #[cfg(feature = "video")]
            // .field("nf_skip", &self.nf_skip)
            .field("media_type", &self.media_type)
            .field("bound", &self.bound)
            .field("progress_bar", &self.with_progress_bar)
            .finish()
    }
}

impl FromStr for DataLoader {
    type Err = anyhow::Error;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        Self::new(source)
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        // paths & media_type
        let (paths, media_type) = Self::try_load_all(source)?;

        // Number of frames or stream
        #[cfg(feature = "video")]
        let mut nf = match media_type {
            MediaType::Image(Location::Local) => {
                paths.as_ref().unwrap_or(&VecDeque::new()).len() as _
            }
            MediaType::Image(Location::Remote) | MediaType::Video(_) | MediaType::Stream => 1,
            MediaType::Unknown => anyhow::bail!("Could not locate the source: {:?}", source),
            _ => unimplemented!(),
        };
        #[cfg(not(feature = "video"))]
        let nf = match media_type {
            MediaType::Image(Location::Local) => {
                paths.as_ref().unwrap_or(&VecDeque::new()).len() as _
            }
            MediaType::Image(Location::Remote) | MediaType::Video(_) | MediaType::Stream => 1,
            MediaType::Unknown => anyhow::bail!("Could not locate the source: {:?}", source),
            _ => unimplemented!(),
        };

        // video decoder
        #[cfg(not(feature = "video"))]
        {
            match &media_type {
                MediaType::Video(Location::Local)
                | MediaType::Video(Location::Remote)
                | MediaType::Stream => {
                    anyhow::bail!(
                        "Video processing requires the features: `video`. \
                        \nConsider enabling them by passing, e.g., `--features video`"
                    );
                }
                _ => {}
            };
        }
        #[cfg(feature = "video")]
        let decoder = match &media_type {
            MediaType::Video(Location::Local) => Some(Decoder::new(Path::new(source))?),
            MediaType::Video(Location::Remote) | MediaType::Stream => {
                let location: video_rs::location::Location = source.parse::<Url>()?.into();
                Some(Decoder::new(location)?)
            }
            _ => None,
        };

        // video & stream frames
        #[cfg(feature = "video")]
        let mut frame_rate = 0.0;

        #[cfg(feature = "video")]
        if let Some(decoder) = &decoder {
            nf = match decoder.frames() {
                Err(_) => u64::MAX,
                Ok(0) => u64::MAX,
                Ok(x) => x,
            };
            frame_rate = decoder.frame_rate();
        }

        // info
        let info = match &media_type {
            MediaType::Image(_) => format!("x{}", nf),
            MediaType::Video(_) => format!("x1 ({} frames)", nf),
            MediaType::Stream => "x1".to_string(),
            _ => unimplemented!(),
        };
        info!("Found {:?} {}", media_type, info);

        Ok(Self {
            paths,
            media_type,
            nf,
            #[cfg(feature = "video")]
            frame_rate,
            #[cfg(feature = "video")]
            decoder,
            ..Default::default()
        })
    }

    pub fn build(mut self) -> Result<Self> {
        let (sender, receiver) =
            mpsc::sync_channel::<Vec<Image>>(self.bound.unwrap_or(self.batch_size * 10));
        self.receiver = receiver;
        let batch_size = self.batch_size;
        #[cfg(feature = "video")]
        let nf_skip = self.nf_skip;
        let data = self.paths.take().unwrap_or_default();
        let media_type = self.media_type;
        #[cfg(feature = "video")]
        let decoder = self.decoder.take();

        // progress bar
        self.progress_bar = if self.with_progress_bar {
            crate::build_progress_bar(
                self.nf,
                "Iterating",
                Some(&format!("{:?}", self.media_type)),
                "{prefix:>12.cyan.bold} {human_pos}/{human_len} |{bar}| {msg}",
            )
            .ok()
        } else {
            None
        };

        // Spawn the producer thread
        std::thread::spawn(move || {
            DataLoader::producer_thread(
                sender,
                data,
                batch_size,
                #[cfg(feature = "video")]
                nf_skip,
                media_type,
                #[cfg(feature = "video")]
                decoder,
            );
        });

        Ok(self)
    }

    fn producer_thread(
        sender: mpsc::SyncSender<Vec<Image>>,
        mut data: VecDeque<PathBuf>,
        batch_size: usize,
        #[cfg(feature = "video")] nf_skip: u64,
        media_type: MediaType,
        #[cfg(feature = "video")] mut decoder: Option<video_rs::decode::Decoder>,
    ) {
        let mut images: Vec<Image> = Vec::with_capacity(batch_size);

        match media_type {
            MediaType::Image(_) => {
                if data.len() < 8000 {
                    //  TODO: fast but memory inefficient
                    crate::elapsed_dataloader!("batch_parallel_read", {
                        data.par_iter()
                            .filter_map(|path| {
                                Some(crate::elapsed_dataloader!("single_image_read", {
                                    Self::try_read_one(path)
                                        .map_err(|e| warn!("Failed: {:?}, {}", path, e))
                                        .ok()?
                                        .with_media_type(media_type)
                                }))
                            })
                            .collect::<Vec<Image>>()
                            .chunks(batch_size)
                            .for_each(|chunk| {
                                if !chunk.is_empty() {
                                    let _ = sender.send(chunk.to_vec());
                                }
                            })
                    });
                } else {
                    // TODO: slow slow
                    crate::elapsed_dataloader!("sequential_read", {
                        while let Some(path) = data.pop_front() {
                            match crate::elapsed_dataloader!("single_image_read", {
                                Self::try_read_one(&path)
                            }) {
                                Err(_err) => {
                                    continue;
                                }
                                Ok(img) => {
                                    images.push(img.with_media_type(media_type));
                                }
                            }
                            if images.len() == batch_size
                                && sender.send(std::mem::take(&mut images)).is_err()
                            {
                                break;
                            }
                        }
                    });
                }
            }
            #[cfg(feature = "video")]
            MediaType::Video(_) | MediaType::Stream => {
                if let Some(decoder) = decoder.as_mut() {
                    let (w, h) = decoder.size();
                    let mut cnt = 0;

                    for frame in decoder.decode_iter() {
                        match frame {
                            Ok((ts, frame)) => {
                                cnt += 1;
                                if (cnt - 1) % (nf_skip + 1) != 0 {
                                    continue;
                                }

                                let rgb8: image::RgbImage = match image::ImageBuffer::from_raw(
                                    w as _,
                                    h as _,
                                    frame.into_raw_vec_and_offset().0,
                                ) {
                                    Some(x) => x,
                                    None => continue,
                                };

                                images.push(
                                    Image::from(rgb8)
                                        .with_media_type(media_type)
                                        .with_source(format!("{:?}", ts).into()),
                                );

                                if images.len() == batch_size
                                    && sender.send(std::mem::take(&mut images)).is_err()
                                {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
            }
            _ => unimplemented!(),
        }

        // Deal with remaining data
        if !images.is_empty() && sender.send(images).is_err() {
            info!("Receiver dropped, stopping production");
        }
    }

    pub fn imread<P: AsRef<Path>>(path: P) -> Result<Image> {
        Image::try_read(path)
    }

    pub fn try_read_one<P: AsRef<Path>>(path: P) -> Result<Image> {
        crate::elapsed_dataloader!("image_decode", Image::try_read(path))
    }

    pub fn try_read_n<P: AsRef<Path> + std::fmt::Debug + Sync>(paths: &[P]) -> Result<Vec<Image>> {
        let images: Vec<Image> = crate::elapsed_dataloader!("batch_read_n", {
            paths
                .par_iter()
                .filter_map(|path| match Self::try_read_one(path) {
                    Ok(img) => Some(img),
                    Err(err) => {
                        log::warn!("Failed to read from: {:?}. Error: {:?}", path, err);
                        None
                    }
                })
                .collect()
        });

        Ok(images)
    }

    pub fn try_read_folder<P: AsRef<Path>>(path: P) -> Result<Vec<Image>> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 path: {:?}", path.as_ref()))?;
        let paths: Vec<PathBuf> = crate::elapsed_dataloader!("folder_scan", {
            Self::load_image_paths_from_folder(path_str, crate::IMAGE_EXTENSIONS)?
        });
        let images: Vec<Image> = crate::elapsed_dataloader!("folder_read", {
            paths
                .par_iter()
                .filter_map(|path| Self::try_read_one(path).ok())
                .collect()
        });

        Ok(images)
    }

    pub fn try_read_pattern(path: &str) -> Result<Vec<Image>> {
        // case sensitive
        let paths: Vec<PathBuf> = Self::glob(path, true, true)?;
        let images: Vec<Image> = paths
            .par_iter()
            .filter_map(|path| Self::try_read_one(path).ok())
            .collect();

        Ok(images)
    }

    pub fn try_read_pattern_case_insensitive(path: &str) -> Result<Vec<Image>> {
        // case insensitive
        let paths: Vec<PathBuf> = Self::glob(path, true, false)?;
        let images: Vec<Image> = paths
            .par_iter()
            .filter_map(|path| Self::try_read_one(path).ok())
            .collect();

        Ok(images)
    }

    fn load_image_paths_from_folder(source: &str, exts: &[&str]) -> Result<Vec<PathBuf>> {
        let source_path = Path::new(source);
        let mut paths: Vec<PathBuf> = Vec::new();
        let config = MatchOptions {
            case_sensitive: false,
            require_literal_separator: false,
            require_literal_leading_dot: false,
        };
        for ext in exts.iter() {
            let pattern = source_path.join(format!("*.{}", ext));
            let pattern_str = pattern
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 pattern path: {:?}", pattern))?;
            let paths_: Vec<PathBuf> = glob_with(pattern_str, config)?
                .filter_map(|entry| entry.ok())
                .collect();
            paths.extend(paths_);
        }

        paths.sort_by(|a, b| {
            let a = a.file_name().and_then(|s| s.to_str());
            let b = b.file_name().and_then(|s| s.to_str());
            match (a, b) {
                (Some(a), Some(b)) => natord::compare(a, b),
                _ => std::cmp::Ordering::Equal,
            }
        });

        Ok(paths)
    }

    fn glob(pattern: &str, sort: bool, case_sensitive: bool) -> anyhow::Result<Vec<PathBuf>> {
        let config = MatchOptions {
            case_sensitive,
            require_literal_separator: false,
            require_literal_leading_dot: false,
        };
        let mut paths: Vec<PathBuf> = glob_with(pattern, config)?
            .filter_map(|entry| entry.ok())
            .collect();

        if sort {
            paths.sort_by(|a, b| {
                let a = a.file_name().and_then(|s| s.to_str());
                let b = b.file_name().and_then(|s| s.to_str());
                match (a, b) {
                    (Some(a), Some(b)) => natord::compare(a, b),
                    _ => std::cmp::Ordering::Equal,
                }
            });
        }

        Ok(paths)
    }

    fn try_load_all(source: &str) -> Result<(Option<VecDeque<PathBuf>>, MediaType)> {
        // paths & media_type
        let is_source_remote = MediaType::is_possible_remote(source);
        let source_path = Path::new(source);
        let (paths, media_type) = if is_source_remote {
            // remote
            log::debug!("DataLoader try to load source from remote");
            (
                Some(VecDeque::from([source_path.to_path_buf()])),
                MediaType::from_url(source),
            )
        } else {
            // local
            log::debug!("DataLoader try to load source from local");
            if source_path.is_file() {
                log::debug!("source is file");
                // image
                (
                    Some(VecDeque::from([source_path.to_path_buf()])),
                    MediaType::from_path(source_path),
                )
            } else if source_path.is_dir() {
                // directory
                log::debug!("source is directory");
                let paths = Self::load_image_paths_from_folder(source, crate::IMAGE_EXTENSIONS)?;

                (
                    Some(VecDeque::from(paths)),
                    MediaType::Image(Location::Local),
                )
            } else if glob::Pattern::new(source).is_ok() {
                log::debug!("Load source with glob pattern");
                // glob
                // - case_sensitive: true
                // - sort: true
                let paths = Self::glob(source, true, true)?;

                (
                    Some(VecDeque::from(paths)),
                    MediaType::Image(Location::Local),
                )
            } else {
                log::debug!("Source is unknown");
                (None, MediaType::Unknown)
            }
        };

        Ok((paths, media_type))
    }
    pub fn paths(&self) -> Option<&VecDeque<PathBuf>> {
        self.paths.as_ref()
    }

    pub fn with_bound(mut self, x: usize) -> Self {
        self.bound = Some(x);
        self
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn with_batch_size_all(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn nf(&self) -> u64 {
        self.nf
    }
    #[cfg(feature = "video")]
    pub fn with_nf_skip(mut self, x: u64) -> Self {
        self.nf_skip = x;
        self
    }

    #[cfg(feature = "video")]
    pub fn nf_skip(&self) -> u64 {
        self.nf_skip
    }

    #[cfg(feature = "video")]
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }

    pub fn with_progress_bar(mut self, x: bool) -> Self {
        self.with_progress_bar = x;
        self
    }

    pub fn iter(&self) -> DataLoaderIter<'_> {
        DataLoaderIter {
            receiver: &self.receiver,
            progress_bar: self.progress_bar.as_ref(),
            batch_size: self.batch_size as u64,
        }
    }
}

trait DataLoaderIterator {
    type Receiver;

    fn receiver(&self) -> &Self::Receiver;
    fn batch_size(&self) -> u64;

    fn progress_bar(&self) -> Option<&ProgressBar>;

    fn next_impl(
        &mut self,
        recv_result: Result<Vec<Image>, mpsc::RecvError>,
    ) -> Option<Vec<Image>> {
        match self.progress_bar() {
            Some(progress_bar) => match recv_result {
                Ok(item) => {
                    progress_bar.inc(self.batch_size());
                    Some(item)
                }
                Err(_) => {
                    progress_bar.set_prefix("Iterated");
                    progress_bar.set_style(
                        ProgressStyle::with_template(
                            crate::PROGRESS_BAR_STYLE_FINISH_2, // "{prefix:>12.green.bold} {msg} x{human_len} in {elapsed}",
                        )
                        .map_err(|e| anyhow!("Style error: {}", e))
                        .ok()?,
                    );
                    progress_bar.finish();
                    None
                }
            },
            None => recv_result.ok(),
        }
    }
}

/// An iterator implementation for `DataLoader` that enables batch processing of images.
///
/// This struct is created by the `into_iter` method on `DataLoader`.
/// It provides functionality for:
/// - Receiving batches of images through a channel
/// - Tracking progress with an optional progress bar
/// - Processing images in configurable batch sizes
pub struct DataLoaderIntoIterator {
    /// Channel receiver for getting batches of images
    receiver: mpsc::Receiver<Vec<Image>>,
    /// Optional progress bar for tracking iteration progress
    progress_bar: Option<ProgressBar>,
    /// Number of images to process in each batch
    batch_size: u64,
}

impl DataLoaderIterator for DataLoaderIntoIterator {
    type Receiver = mpsc::Receiver<Vec<Image>>;

    fn receiver(&self) -> &Self::Receiver {
        &self.receiver
    }

    fn batch_size(&self) -> u64 {
        self.batch_size
    }

    fn progress_bar(&self) -> Option<&ProgressBar> {
        self.progress_bar.as_ref()
    }
}

impl Iterator for DataLoaderIntoIterator {
    type Item = Vec<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl(self.receiver().recv())
    }
}

impl IntoIterator for DataLoader {
    type Item = Vec<Image>;
    type IntoIter = DataLoaderIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        DataLoaderIntoIterator {
            receiver: self.receiver,
            progress_bar: self.progress_bar,
            batch_size: self.batch_size as u64,
        }
    }
}

/// A borrowing iterator for `DataLoader` that enables batch processing of images.
///
/// This iterator is created by the `iter()` method on `DataLoader`, allowing iteration
/// over batches of images without taking ownership of the `DataLoader`.
///
/// # Fields
/// - `receiver`: A reference to the channel receiver that provides batches of images
/// - `progress_bar`: An optional reference to a progress bar for tracking iteration progress
/// - `batch_size`: The number of images to process in each batch
pub struct DataLoaderIter<'a> {
    receiver: &'a mpsc::Receiver<Vec<Image>>,
    progress_bar: Option<&'a ProgressBar>,
    batch_size: u64,
}

impl DataLoaderIterator for DataLoaderIter<'_> {
    type Receiver = mpsc::Receiver<Vec<Image>>;

    fn receiver(&self) -> &Self::Receiver {
        self.receiver
    }

    fn batch_size(&self) -> u64 {
        self.batch_size
    }

    fn progress_bar(&self) -> Option<&ProgressBar> {
        self.progress_bar
    }
}

impl Iterator for DataLoaderIter<'_> {
    type Item = Vec<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl(self.receiver().recv())
    }
}

impl<'a> IntoIterator for &'a DataLoader {
    type Item = Vec<Image>;
    type IntoIter = DataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DataLoaderIter {
            receiver: &self.receiver,
            progress_bar: self.progress_bar.as_ref(),
            batch_size: self.batch_size as u64,
        }
    }
}
