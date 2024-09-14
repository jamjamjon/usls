use anyhow::{anyhow, Result};
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use video_rs::{
    encode::{Encoder, Settings},
    time::Time,
    Decoder, Url,
};

use crate::{
    build_progress_bar, string_now, Dir, Hub, Location, MediaType, CHECK_MARK, CROSS_MARK,
};

type TempReturnType = (Vec<DynamicImage>, Vec<PathBuf>);

pub struct DataLoaderIterator {
    receiver: mpsc::Receiver<TempReturnType>,
    progress_bar: Option<ProgressBar>,
}

impl Iterator for DataLoaderIterator {
    type Item = TempReturnType;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.progress_bar {
            None => self.receiver.recv().ok(),
            Some(progress_bar) => {
                progress_bar.inc(1);
                match self.receiver.recv().ok() {
                    Some(item) => {
                        // progress_bar.inc(1);
                        Some(item)
                    }
                    None => {
                        progress_bar.set_prefix("    Iterated");
                        progress_bar.finish();
                        None
                    }
                }
            }
        }
    }
}

impl IntoIterator for DataLoader {
    type Item = TempReturnType;
    type IntoIter = DataLoaderIterator;

    fn into_iter(self) -> Self::IntoIter {
        let progress_bar = if self.with_pb {
            build_progress_bar(
                self.nf / self.batch_size as u64,
                "   Iterating",
                Some(&format!("{:?}", self.media_type)),
                "{prefix:.green.bold} {msg} {human_pos}/{human_len} |{bar}| {elapsed_precise}",
            )
            .ok()
        } else {
            None
        };

        DataLoaderIterator {
            receiver: self.receiver,
            progress_bar,
        }
    }
}

/// Load images, video, stream
pub struct DataLoader {
    pub paths: Option<VecDeque<PathBuf>>,
    pub media_type: MediaType,
    pub batch_size: usize,
    sender: Option<mpsc::Sender<TempReturnType>>,
    receiver: mpsc::Receiver<TempReturnType>,
    pub decoder: Option<video_rs::decode::Decoder>,
    nf: u64, // MAX means live stream
    with_pb: bool,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            paths: None,
            media_type: MediaType::Unknown,
            batch_size: 1,
            sender: None,
            receiver: mpsc::channel().1,
            decoder: None,
            nf: 0,
            with_pb: true,
        }
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        let span = tracing::span!(tracing::Level::INFO, "DataLoader-new");
        let _guard = span.enter();

        // Number of frames or stream
        let mut nf = 0;

        // paths & media_type
        let source_path = Path::new(source);
        let (paths, media_type) = match source_path.exists() {
            false => {
                // remote
                nf = 1;
                (
                    Some(VecDeque::from([source_path.to_path_buf()])),
                    MediaType::from_url(source),
                )
            }
            true => {
                // local
                if source_path.is_file() {
                    nf = 1;
                    (
                        Some(VecDeque::from([source_path.to_path_buf()])),
                        MediaType::from_path(source_path),
                    )
                } else if source_path.is_dir() {
                    let paths_sorted = Self::load_from_folder(source_path)?;
                    nf = paths_sorted.len() as _;
                    (
                        Some(VecDeque::from(paths_sorted)),
                        MediaType::Image(Location::Local),
                    )
                } else {
                    (None, MediaType::Unknown)
                }
            }
        };

        if let MediaType::Unknown = media_type {
            anyhow::bail!("Could not locate the source path: {:?}", source_path);
        }

        // mpsc
        let (sender, receiver) = mpsc::channel::<TempReturnType>();

        // decoder
        let decoder = match &media_type {
            MediaType::Video(Location::Local) => Some(Decoder::new(source_path)?),
            MediaType::Video(Location::Remote) | MediaType::Stream => {
                let location: video_rs::location::Location = source.parse::<Url>()?.into();
                Some(Decoder::new(location)?)
            }
            _ => None,
        };

        // get frames
        if let Some(decoder) = &decoder {
            nf = decoder.frames().unwrap_or(u64::MAX);
        }

        // summary
        tracing::info!("{} Found {:?} x{}", CHECK_MARK, media_type, nf,);

        Ok(DataLoader {
            paths,
            media_type,
            sender: Some(sender),
            receiver,
            batch_size: 1,
            decoder,
            nf,
            with_pb: true,
        })
    }

    pub fn build(mut self) -> Result<Self> {
        let sender = self.sender.take().expect("Sender should be available");
        let batch_size = self.batch_size;
        let data = self.paths.take().unwrap_or_default();
        // let media_type = self.media_type.take().unwrap_or(MediaType::Unknown);
        let media_type = self.media_type.clone();
        let decoder = self.decoder.take();

        // Spawn the producer thread
        std::thread::spawn(move || {
            DataLoader::producer_thread(sender, data, batch_size, media_type, decoder);
        });

        Ok(self)
    }

    fn producer_thread(
        sender: mpsc::Sender<TempReturnType>,
        mut data: VecDeque<PathBuf>,
        batch_size: usize,
        media_type: MediaType,
        mut decoder: Option<video_rs::decode::Decoder>,
    ) {
        let span = tracing::span!(tracing::Level::INFO, "DataLoader-producer-thread");
        let _guard = span.enter();
        let mut yis: Vec<DynamicImage> = Vec::with_capacity(batch_size);
        let mut yps: Vec<PathBuf> = Vec::with_capacity(batch_size);

        match media_type {
            MediaType::Image(_) => {
                while let Some(path) = data.pop_front() {
                    match Self::try_read(&path) {
                        Err(err) => {
                            tracing::warn!("{} {:?} | {:?}", CROSS_MARK, path, err);
                            continue;
                        }
                        Ok(img) => {
                            yis.push(img);
                            yps.push(path.clone());
                        }
                    }
                    if yis.len() == batch_size
                        && sender
                            .send((std::mem::take(&mut yis), std::mem::take(&mut yps)))
                            .is_err()
                    {
                        break;
                    }
                }
            }
            MediaType::Video(_) | MediaType::Stream => {
                if let Some(decoder) = decoder.as_mut() {
                    let (w, h) = decoder.size();
                    let frames = decoder.decode_iter();

                    for frame in frames {
                        match frame {
                            Ok((ts, frame)) => {
                                let rgb8: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                                    match image::ImageBuffer::from_raw(
                                        w as _,
                                        h as _,
                                        frame.into_raw_vec_and_offset().0,
                                    ) {
                                        Some(x) => x,
                                        None => continue,
                                    };
                                let img = image::DynamicImage::from(rgb8);
                                yis.push(img);
                                yps.push(ts.to_string().into());

                                if yis.len() == batch_size
                                    && sender
                                        .send((std::mem::take(&mut yis), std::mem::take(&mut yps)))
                                        .is_err()
                                {
                                    break;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                }
            }
            _ => todo!(),
        }

        // Deal with remaining data
        if !yis.is_empty() && sender.send((yis, yps)).is_err() {
            tracing::info!("Receiver dropped, stopping production");
        }
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn with_progress_bar(mut self, x: bool) -> Self {
        self.with_pb = x;
        self
    }

    pub fn load_from_folder<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<std::path::PathBuf>> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        paths.sort_by(|a, b| {
            let a_name = a.file_name().and_then(|s| s.to_str());
            let b_name = b.file_name().and_then(|s| s.to_str());

            match (a_name, b_name) {
                (Some(a_str), Some(b_str)) => natord::compare(a_str, b_str),
                _ => std::cmp::Ordering::Equal,
            }
        });

        Ok(paths)
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let mut path = path.as_ref().to_path_buf();

        // try to fetch from hub or local cache
        if !path.exists() {
            let p = Hub::new()?.fetch(path.to_str().unwrap())?.commit()?;
            path = PathBuf::from(&p);
        }
        let img = Self::read_into_rgb8(path)?;
        Ok(DynamicImage::from(img))
    }

    fn read_into_rgb8<P: AsRef<Path>>(path: P) -> Result<image::RgbImage> {
        let path = path.as_ref();
        let img = image::ImageReader::open(path)
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
        Ok(img)
    }

    /// Convert images into a video
    pub fn is2v<P: AsRef<Path>>(source: P, subs: &[&str], fps: usize) -> Result<()> {
        let paths = Self::load_from_folder(source.as_ref())?;
        if paths.is_empty() {
            anyhow::bail!("No images found.");
        }
        let mut encoder = None;
        let mut position = Time::zero();
        let saveout = Dir::Currnet
            .raw_path_with_subs(subs)?
            .join(format!("{}.mp4", string_now("-")));
        let pb = build_progress_bar(paths.len() as u64, "  Converting", saveout.to_str(),"{prefix:.cyan.bold} {msg} |{bar}| ({percent_precise}%, {human_pos}/{human_len}, {per_sec})")?;

        // loop
        for path in paths {
            pb.inc(1);
            let img = Self::read_into_rgb8(path)?;
            let (w, h) = img.dimensions();

            // build encoder at the 1st time
            if encoder.is_none() {
                let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
                encoder = Some(Encoder::new(saveout.clone(), settings)?);
            }

            // write video
            if let Some(encoder) = encoder.as_mut() {
                let raw_data = img.into_raw();
                let frame = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), raw_data)
                    .expect("Failed to create ndarray from raw image data");

                // encode and update
                encoder.encode(&frame, position)?;
                position = position.aligned_with(Time::from_nth_of_a_second(fps)).add();
            }
        }

        match &mut encoder {
            Some(vencoder) => vencoder.finish()?,
            None => anyhow::bail!("Found no video encoder."),
        }

        // update
        pb.set_prefix("  Downloaded");
        pb.set_prefix("   Converted");
        pb.set_style(ProgressStyle::with_template(
            "{prefix:.green.bold} {msg} in {elapsed}",
        )?);
        pb.finish();

        Ok(())
    }
}
