use anyhow::{anyhow, Result};
use image::DynamicImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use video_rs::{Decoder, Url};
use walkdir::{DirEntry, WalkDir}; // TODO: remove

use crate::{Dir, Hub, Location, MediaType, CHECK_MARK};

type TempReturnType = (Vec<DynamicImage>, Vec<PathBuf>);

/// Dataloader for loading images, videos and streams,
// #[derive(Debug, Clone)]
pub struct DataLoader {
    // source
    // - image(local & remote)
    // - images(dir)
    // - video(local & remote)
    // - stream(local & remote)
    pub paths: VecDeque<PathBuf>,
    pub media_type: MediaType,
    pub recursive: bool, // TODO: remove
    pub batch_size: usize,
    pub buffer_count: Arc<AtomicUsize>,
    sender: Option<mpsc::Sender<TempReturnType>>,
    receiver: mpsc::Receiver<TempReturnType>,
    pub decoder: Option<video_rs::decode::Decoder>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self {
            paths: VecDeque::new(),
            media_type: MediaType::Unknown,
            recursive: false,
            batch_size: 1,
            buffer_count: Arc::new(AtomicUsize::new(0)),
            sender: None,
            receiver: mpsc::channel().1,
            decoder: None,
        }
    }
}

impl DataLoader {
    pub fn new(source: &str) -> Result<Self> {
        let mut paths = VecDeque::new();

        // local or remote
        let source_path = Path::new(source);
        let media_type = if source_path.exists() {
            // local
            if source_path.is_file() {
                paths.push_back(source_path.to_path_buf());
                MediaType::from_path(source_path)
            } else if source_path.is_dir() {
                // dir => only can be images
                for entry in (source_path.read_dir().map_err(|e| e.to_string()).unwrap()).flatten()
                {
                    if entry.path().is_file() {
                        paths.push_back(entry.path());
                    }
                }
                MediaType::Image(Location::Local)
            } else {
                MediaType::Unknown
            }
        } else {
            // remote
            paths.push_back(PathBuf::from(source));
            MediaType::from_url(source)
        };

        let (sender, receiver) = mpsc::channel::<TempReturnType>();
        let buffer_count = Arc::new(AtomicUsize::new(0));

        // decoder
        let decoder = match &media_type {
            MediaType::Video(Location::Local) => {
                let location: video_rs::location::Location = paths[0].clone().into();
                Some(Decoder::new(location).unwrap())
            }
            MediaType::Video(Location::Remote) | MediaType::Stream => {
                let location: video_rs::location::Location =
                    paths[0].to_str().unwrap().parse::<Url>().unwrap().into();

                Some(Decoder::new(location).unwrap())
            }
            _ => None,
        };

        println!("{CHECK_MARK} Media Type: {:?} x{}", media_type, paths.len());
        Ok(DataLoader {
            paths,
            media_type,
            buffer_count,
            sender: Some(sender),
            receiver,
            recursive: false,
            batch_size: 1,
            decoder,
        })
    }

    // Initialize the producer thread
    pub fn commit(&mut self) {
        let sender = self.sender.take().expect("Sender should be available");
        let buffer_count = Arc::clone(&self.buffer_count);
        let batch_size = self.batch_size;
        let data = self.paths.clone();
        let media_type = self.media_type.clone();
        let decoder = self.decoder.take();

        std::thread::spawn(move || {
            DataLoader::producer_thread(
                sender,
                buffer_count,
                data,
                batch_size,
                media_type,
                decoder,
            );
        });
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer_count.load(Ordering::SeqCst)
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
        self.batch_size = x;
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

    fn producer_thread(
        sender: mpsc::Sender<TempReturnType>,
        buffer_count: Arc<AtomicUsize>,
        mut data: VecDeque<PathBuf>,
        batch_size: usize,
        media_type: MediaType,
        mut decoder: Option<video_rs::decode::Decoder>,
    ) {
        let mut yis: Vec<DynamicImage> = Vec::new();
        let mut yps: Vec<PathBuf> = Vec::new();

        match media_type {
            MediaType::Image(Location::Local) => {
                while let Some(path) = data.pop_front() {
                    match Self::try_read(&path) {
                        Err(err) => {
                            println!("Error reading image from path {:?}: {:?}", path, err);
                            continue;
                        }
                        Ok(img) => {
                            yis.push(img);
                            yps.push(path.clone());
                            buffer_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }

                    if yis.len() == batch_size
                        && sender
                            .send((std::mem::take(&mut yis), std::mem::take(&mut yps)))
                            .is_err()
                    {
                        println!("Receiver dropped, stopping production");
                        break;
                    }
                }
            }
            MediaType::Image(Location::Remote) => {
                while let Some(path) = data.pop_front() {
                    let file_name = path.file_name().unwrap();
                    let p_tmp = Dir::Cache.path(Some("tmp")).unwrap().join(file_name);
                    Hub::download(path.to_str().unwrap(), &p_tmp, None, None, None).unwrap();
                    match Self::try_read(&p_tmp) {
                        Err(err) => {
                            println!(
                                "Error reading downloaded image from path {:?}: {:?}",
                                p_tmp, err
                            );
                            continue;
                        }
                        Ok(x) => {
                            yis.push(x);
                            yps.push(path.clone());
                            buffer_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }

                    if yis.len() == batch_size
                        && sender
                            .send((std::mem::take(&mut yis), std::mem::take(&mut yps)))
                            .is_err()
                    {
                        println!("Receiver dropped, stopping production");
                        break;
                    }
                }
            }
            MediaType::Video(_) => {
                if let Some(decoder) = decoder.as_mut() {
                    let (w, h) = decoder.size();
                    let frames = decoder.decode_iter();

                    // while let Some(frame) = frames.next() {
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
                                buffer_count.fetch_add(1, Ordering::SeqCst);

                                if yis.len() == batch_size
                                    && sender
                                        .send((std::mem::take(&mut yis), std::mem::take(&mut yps)))
                                        .is_err()
                                {
                                    println!("Receiver dropped, stopping production");
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
}

impl Iterator for DataLoader {
    type Item = (Vec<DynamicImage>, Vec<PathBuf>);

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(batch) => {
                let t0 = std::time::Instant::now();
                self.buffer_count
                    .fetch_sub(self.batch_size, Ordering::SeqCst);
                let t1 = std::time::Instant::now();
                println!("==> {:?}", t1 - t0);
                Some(batch)
            }
            Err(_) => None,
        }
    }
}
