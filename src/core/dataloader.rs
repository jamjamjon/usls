use anyhow::{anyhow, Result};
use image::DynamicImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use video_rs::{Decoder, Url};

use crate::{Hub, Location, MediaType, CHECK_MARK};

type TempReturnType = (Vec<DynamicImage>, Vec<PathBuf>);

/// Dataloader for loading image, video and stream,
pub struct DataLoader {
    // source could be:
    // - image(local & remote(hub))
    // - images(dir)
    // - video(local & remote)
    // - stream(local & remote)
    pub paths: VecDeque<PathBuf>,
    pub media_type: MediaType,
    pub batch_size: usize,
    sender: Option<mpsc::Sender<TempReturnType>>,
    receiver: mpsc::Receiver<TempReturnType>,
    pub decoder: Option<video_rs::decode::Decoder>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            paths: VecDeque::new(),
            media_type: MediaType::Unknown,
            batch_size: 1,
            sender: None,
            receiver: mpsc::channel().1,
            decoder: None,
        }
    }
}

impl Iterator for DataLoader {
    type Item = TempReturnType;

    fn next(&mut self) -> Option<Self::Item> {
        let t0 = std::time::Instant::now();
        match self.receiver.recv() {
            Ok(batch) => {
                let t1 = std::time::Instant::now();
                println!("==> {:?}", t1 - t0);
                Some(batch)
            }
            Err(_) => None,
        }
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        // paths & media_type
        let source_path = Path::new(source);
        let (paths, media_type) = match source_path.exists() {
            false => {
                // remote
                (
                    VecDeque::from([source_path.to_path_buf()]),
                    MediaType::from_url(source),
                )
            }
            true => {
                // local
                if source_path.is_file() {
                    (
                        VecDeque::from([source_path.to_path_buf()]),
                        MediaType::from_path(source_path),
                    )
                } else if source_path.is_dir() {
                    let mut entries: Vec<PathBuf> = std::fs::read_dir(source_path)?
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
                    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                    (VecDeque::from(entries), MediaType::Image(Location::Local))
                } else {
                    (VecDeque::new(), MediaType::Unknown)
                }
            }
        };

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

        // summary
        println!("{CHECK_MARK} Found {:?} x{}", media_type, paths.len());

        Ok(DataLoader {
            paths,
            media_type,
            sender: Some(sender),
            receiver,
            batch_size: 1,
            decoder,
        })
    }

    // Build to initialize the producer thread
    pub fn build(mut self) -> Result<Self> {
        let sender = self.sender.take().expect("Sender should be available");
        let batch_size = self.batch_size;
        let data = self.paths.clone();
        let media_type = self.media_type.clone();
        let decoder = self.decoder.take();

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
        let mut yis: Vec<DynamicImage> = Vec::with_capacity(batch_size);
        let mut yps: Vec<PathBuf> = Vec::with_capacity(batch_size);

        match media_type {
            MediaType::Image(_) => {
                while let Some(path) = data.pop_front() {
                    match Self::try_read(&path) {
                        Err(_) => {
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

        // Send any remaining data
        if !yis.is_empty() && sender.send((yis, yps)).is_err() {
            println!("Receiver dropped, stopping production");
        }
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn try_read<P: AsRef<Path>>(path: P) -> Result<DynamicImage> {
        let mut path = path.as_ref().to_path_buf();

        // try to fetch from hub or local cache
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
}
