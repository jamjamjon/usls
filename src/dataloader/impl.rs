//! DataLoader module for high-performance data loading

use anyhow::Result;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::mpsc;

use crate::{DataLoaderIter, Image, Source, SourceType, PB};

/// A high-performance data loading engine for images, videos, and streams.
///
/// # DataLoader
///
/// Built-in batching and parallel data ingestion engine designed for computer vision
/// and machine learning workloads.
///
/// ## Supported Inputs
///
/// - **image**: Local files, remote URLs, GitHub releases
///   - Examples: `"image.jpg"`, `"https://example.com/img.jpg"`, `"images/bus.jpg"`
/// - **video/stream**: Local files, remote URLs, webcams, RTSP/RTMP streams
///   - Examples: `"video.mp4"`, `"rtsp://..."`, `"0"` (webcam index)
/// - **collection**: Directories, glob patterns, or mixed lists
///   - Examples: `"./assets/"`, `"./images/*.jpg"`, `"./data/**/*"`
/// - **hybrid**: Combine any sources using `Vec` or separators (`,` or `|`)
///   - Examples: `vec!["img1.jpg", "img2.png"]`, `"video.mp4|images/*.jpg"`
///
/// ## Interface
///
/// ### Synchronous (`try_read_*`)
/// Parallel loading for static images. Returns `Vec<Image>` or `Image`.
///
/// - [`try_read_one()`](Self::try_read_one): Read the first valid image
/// - [`try_read_nth(n)`](Self::try_read_nth): Read the Nth valid image (0-indexed)
/// - [`try_read_range(a..b)`](Self::try_read_range): Read a range of images
/// - [`try_read()`](Self::try_read): Read all valid images from mixed sources
///
/// ### Streaming (`stream()`)
/// Background thread for non-blocking ingestion (recommended for all sources).
/// Frames are processed in a background thread and yielded in batches.
///
/// - **Universal**: Works for images, videos, webcams, and live streams
/// - **Features**: Supports frame skipping, progress bars, and batching
/// - **Iterator**: Yields `Vec<Image>` batches via `IntoIterator`
///
/// # Examples
///
/// ## Synchronous Loading
///
/// ```no_run
/// use usls::DataLoader;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Single image
/// let image = DataLoader::new("image.jpg")?.try_read_one()?;
///
/// // Nth image from collection
/// let image = DataLoader::new("./images/*.jpg")?.try_read_nth(2)?;
///
/// // Range of images
/// let images = DataLoader::new("./assets")?.try_read_range(0..5)?;
///
/// // All images from mixed sources
/// let images = DataLoader::new(vec![
///     "local.jpg",
///     "https://example.com/remote.jpg",
///     "./images/*.png",
/// ])?.try_read()?;
/// # Ok(())
/// # }
/// ```
///
/// ## Streaming
///
/// ```no_run
/// use usls::DataLoader;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let source = "./assets/bus.jpg | ../video.mp4 | ./assets/*.png | 0";
/// let dl = DataLoader::new(source)?
///     .with_batch(32)           // 32 images per batch
///     .with_progress_bar(true)  // Show progress
///     .stream()?;               // Start background thread
///
/// for (i, batch) in dl.into_iter().enumerate() {
///     println!("Batch {}: {} images", i, batch.len());
///     // Process batch...
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Features
///
/// - **Parallel Processing**: Uses Rayon for concurrent image loading
/// - **Memory Efficient**: Configurable channel bounds and batch sizes
/// - **Progress Tracking**: Optional progress bars for long operations
/// - **Error Resilient**: Continues processing when individual items fail
/// - **Type Agnostic**: Works with any image format supported by the backend
///
/// # Feature Flags
///
/// - `video`: Enable video and stream support (webcam, RTSP, RTMP)
///
/// # Performance
///
/// - Synchronous methods are optimized for static image collections
/// - Streaming is recommended for videos, webcams, and large datasets
/// - Batch size affects memory usage vs. processing throughput
/// - Channel bounds control memory usage for streaming workloads
///
/// > ⚠️ Video/Stream support requires the `video` feature flag.
pub struct DataLoader {
    /// Data source.
    pub(crate) source: Source,

    /// Batch size for iteration, determining how many files are processed at once.
    pub(crate) batch_size: usize,

    /// Buffer size for the channel, used to manage the buffer between producer and consumer.
    pub(crate) bound: Option<usize>,

    /// Receiver for processed data.
    pub(crate) receiver: mpsc::Receiver<Vec<Image>>,

    /// Total number of yielded items (images or frames).
    pub(crate) nf: u64,

    /// Number of static images.
    pub(crate) nfi: u64,

    /// Number of video frames (before skip).
    pub(crate) nfv: u64,

    /// Number of video frames to be skipped.
    #[cfg(feature = "video")]
    pub(crate) nfv_skip: u64,

    /// Progress bar for displaying iteration progress.
    pub(crate) progress_bar: Option<PB>,

    /// Display progress bar or not.
    pub(crate) with_progress_bar: bool,
}

impl Default for DataLoader {
    fn default() -> Self {
        DataLoader {
            source: Source::new(),
            nf: 0,
            nfi: 0,
            nfv: 0,
            batch_size: 1,
            #[cfg(feature = "video")]
            nfv_skip: 0,
            bound: None,
            receiver: mpsc::sync_channel(0).1,
            progress_bar: None,
            with_progress_bar: true,
        }
    }
}

impl std::fmt::Debug for DataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoader")
            .field("source", &self.source)
            .field("bound", &self.bound)
            .field("batch_size", &self.batch_size)
            .field("num_all_frames", &self.nf)
            .field("num_frames_image", &self.nfi)
            .field("num_frames_video", &self.nfv)
            .finish()
    }
}

impl TryFrom<&str> for DataLoader {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::new(s)
    }
}

impl DataLoader {
    /// Create DataLoader from a unified source.
    pub fn new<S: Into<Source>>(source: S) -> Result<Self> {
        crate::perf!("DataLoader::new", {
            let source = source.into();

            let mut nfi = 0;
            #[allow(unused_mut)]
            let mut nfv = 0;

            for task in source.tasks() {
                match task {
                    SourceType::Image(_)
                    | SourceType::DynamicImage(_)
                    | SourceType::ImageUrl(_) => {
                        nfi += 1;
                    }
                    #[cfg(not(feature = "video"))]
                    SourceType::Video(x) => {
                        anyhow::bail!(
                            "Video support requires the `video` feature flag. Source: {x:?}",
                        )
                    }
                    #[cfg(not(feature = "video"))]
                    SourceType::Stream(x) => {
                        anyhow::bail!(
                            "Stream support requires the `video` feature flag. Source: {x:?}",
                        )
                    }
                    #[cfg(not(feature = "video"))]
                    SourceType::Webcam(x) => {
                        anyhow::bail!(
                            "Webcam support requires the `video` feature flag. Source: {x:?}",
                        )
                    }
                    #[cfg(feature = "video")]
                    SourceType::Video(p) => {
                        if let Ok(decoder) = video_rs::decode::Decoder::new(p.clone()) {
                            match decoder.frames() {
                                Ok(0) => {
                                    tracing::warn!(
                                    "Video decoder returned 0 frames for {p:?}, treating as stream",
                                );
                                    nfv = u64::MAX;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                    "Failed to get frame count for {p:?}: {e}, treating as stream",
                                );
                                    nfv = u64::MAX;
                                }
                                Ok(x) => {
                                    tracing::info!("Video {p:?} has {x} frames");
                                    if nfv != u64::MAX {
                                        nfv += x;
                                    }
                                }
                            }
                        } else {
                            tracing::warn!("Failed to initialize video decoder for {p:?}, treating as 1 static item");
                            nfv += 1;
                        }
                    }
                    #[cfg(feature = "video")]
                    SourceType::Stream(_) | SourceType::Webcam(_) => {
                        nfv = u64::MAX;
                    }
                    _ => {}
                }
            }

            let nf = if nfv == u64::MAX { u64::MAX } else { nfi + nfv };

            tracing::info!(
                "Found {} tasks, total estimated items: {} (images: {}, video frames: {})",
                source.tasks().len(),
                if nf == u64::MAX {
                    "Infinity".to_string()
                } else {
                    nf.to_string()
                },
                nfi,
                if nfv == u64::MAX {
                    "Infinity".to_string()
                } else {
                    nfv.to_string()
                }
            );

            Ok(Self {
                source,
                nf,
                nfi,
                nfv,
                ..Default::default()
            })
        })
    }

    /// Read all data from the source synchronously and in parallel where possible.
    pub fn try_read(&self) -> Result<Vec<Image>> {
        self.try_read_range(..)
    }

    /// Read the first image from the source.
    pub fn try_read_one(&self) -> Result<Image> {
        self.try_read_nth(0)
    }

    fn read_image_based_on_source(task: &SourceType) -> Option<Image> {
        crate::perf!("DataLoader::read-image", {
            match task {
                SourceType::Image(p) => {
                    crate::perf!("DataLoader::read-image::pathbuf", Image::try_read(p).ok())
                }
                SourceType::ImageUrl(url) => {
                    crate::perf!("DataLoader::read-image::url", Image::try_read(url).ok())
                }
                SourceType::DynamicImage(img) => crate::perf!(
                    "DataLoader::read-image::dynamic",
                    Some(Image::from(img.clone()))
                ),
                _ => None,
            }
        })
    }

    /// Load the image at the specified index.
    pub fn try_read_nth(&self, index: usize) -> Result<Image> {
        if self.source.has_video_or_stream() {
            anyhow::bail!("DataLoader::try_read_nth() only supports static image sources.");
        }
        let tasks = self.source.tasks();
        tasks
            .get(index)
            .and_then(Self::read_image_based_on_source)
            .ok_or_else(|| anyhow::anyhow!("No valid image found at index {index}"))
    }

    /// Read a range of images from the source synchronously and in parallel.
    pub fn try_read_range<R>(&self, range: R) -> Result<Vec<Image>>
    where
        R: std::ops::RangeBounds<usize>,
    {
        if self.source.has_video_or_stream() {
            anyhow::bail!("DataLoader::try_read_range() only supports static image sources. For videos/streams, please use .stream() iterator.");
        }

        let len = self.source.len();
        let start = match range.start_bound() {
            std::ops::Bound::Included(&s) => s,
            std::ops::Bound::Excluded(&s) => s + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&e) => (e + 1).min(len),
            std::ops::Bound::Excluded(&e) => e.min(len),
            std::ops::Bound::Unbounded => len,
        };

        if start >= end {
            return Ok(Vec::new());
        }

        let tasks = self.source.tasks();
        let images: Vec<Image> = tasks
            .iter()
            .skip(start)
            .take(end - start)
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter_map(Self::read_image_based_on_source)
            .collect();

        Ok(images)
    }

    /// Build a DataLoader stream iterator.
    pub fn stream(mut self) -> Result<Self> {
        let (sender, receiver) =
            mpsc::sync_channel::<Vec<Image>>(self.bound.unwrap_or(self.batch_size * 10));
        self.receiver = receiver;
        let batch_size = self.batch_size;
        #[cfg(feature = "video")]
        let nfv_skip = self.nfv_skip;

        // Adjust total items (nf) for progress bar if skipping video frames
        if self.nf != u64::MAX {
            #[cfg(feature = "video")]
            {
                let nf_video_skipped = if self.nfv > 0 {
                    (self.nfv + nfv_skip) / (nfv_skip + 1)
                } else {
                    0
                };
                self.nf = self.nfi + nf_video_skipped;
            }
        }

        // progress bar
        if self.with_progress_bar {
            self.progress_bar = Some(PB::iterating(self.nf));
        }

        let tasks = std::mem::take(&mut self.source.tasks);

        // Spawn the producer thread
        std::thread::spawn(move || {
            DataLoader::producer_thread(
                sender,
                tasks,
                batch_size,
                #[cfg(feature = "video")]
                nfv_skip,
            );
        });

        Ok(self)
    }

    fn producer_thread(
        sender: mpsc::SyncSender<Vec<Image>>,
        mut tasks: VecDeque<SourceType>,
        batch_size: usize,
        #[cfg(feature = "video")] nfv_skip: u64,
    ) {
        let mut images: Vec<Image> = Vec::with_capacity(batch_size);

        while !tasks.is_empty() {
            // Streaming Parallel Design:
            // Instead of loading all images into memory at once (the "8000 problem"),
            // we process consecutive static images in small batches (chunks).
            // This maintains high throughput via Rayon while keeping memory usage bounded.
            if tasks.front().is_some_and(|t| t.is_image()) {
                let mut chunk = Vec::new();
                let chunk_limit = std::cmp::max(batch_size * 4, 32);

                while chunk.len() < chunk_limit {
                    if let Some(t) = tasks.front() {
                        if t.is_image() {
                            chunk.push(tasks.pop_front().unwrap());
                            continue;
                        }
                    }
                    break;
                }

                let loaded: Vec<Image> = chunk
                    .into_par_iter()
                    .filter_map(|task| Self::read_image_based_on_source(&task))
                    .collect();

                for img in loaded {
                    images.push(img);
                    if images.len() >= batch_size
                        && sender
                            .send(std::mem::replace(
                                &mut images,
                                Vec::with_capacity(batch_size),
                            ))
                            .is_err()
                    {
                        return;
                    }
                }
            } else if let Some(task) = tasks.pop_front() {
                // Non-image tasks (Video, Stream, Webcam) are processed sequentially
                // as they typically represent continuous temporal data.
                match task {
                    #[cfg(feature = "video")]
                    SourceType::Video(p) => {
                        if let Ok(mut decoder) = video_rs::decode::Decoder::new(p.clone()) {
                            crate::perf!(
                                "DataLoader::decode-video/stream",
                                Self::decode_video(
                                    &mut decoder,
                                    nfv_skip,
                                    batch_size,
                                    &sender,
                                    &mut images,
                                    Some(p),
                                )
                            );
                        }
                    }
                    #[cfg(feature = "video")]
                    SourceType::Stream(url) => {
                        if let Ok(url) = url.parse::<video_rs::Url>() {
                            let source_path = std::path::PathBuf::from(url.as_str());
                            let location: video_rs::location::Location = url.into();
                            if let Ok(mut decoder) = video_rs::decode::Decoder::new(location) {
                                crate::perf!(
                                    "DataLoader::decode-video/stream",
                                    Self::decode_video(
                                        &mut decoder,
                                        nfv_skip,
                                        batch_size,
                                        &sender,
                                        &mut images,
                                        Some(source_path),
                                    )
                                );
                            }
                        }
                    }
                    #[cfg(feature = "video")]
                    SourceType::Webcam(index) => {
                        crate::perf!(
                            "DataLoader::decode-webcam",
                            Self::decode_webcam(index, nfv_skip, batch_size, &sender, &mut images)
                        );
                    }
                    _ => {
                        tracing::error!("Unexpected task type in producer thread: {:?}", task);
                    }
                }
            }

            // Flush remaining images if we switched task types or reached end
            if images.len() >= batch_size
                && sender
                    .send(std::mem::replace(
                        &mut images,
                        Vec::with_capacity(batch_size),
                    ))
                    .is_err()
            {
                return;
            }
        }

        // Final flush
        if !images.is_empty() && sender.send(images).is_err() {
            tracing::info!("Receiver dropped, stopping production");
        }
    }

    #[cfg(feature = "video")]
    fn decode_webcam(
        index: u32,
        nfv_skip: u64,
        batch_size: usize,
        sender: &mpsc::SyncSender<Vec<Image>>,
        images: &mut Vec<Image>,
    ) {
        use video_rs::ffmpeg;

        ffmpeg::init().ok();

        let input_format_name = if cfg!(target_os = "macos") {
            "avfoundation"
        } else if cfg!(target_os = "linux") {
            "video4linux2"
        } else if cfg!(target_os = "windows") {
            "dshow"
        } else {
            tracing::error!("Unsupported OS for webcam");
            return;
        };

        let c_name = std::ffi::CString::new(input_format_name).unwrap();
        let ptr = unsafe { video_rs::ffmpeg::ffi::av_find_input_format(c_name.as_ptr()) };

        let input_format = if ptr.is_null() {
            tracing::error!("Input format '{}' not found", input_format_name);
            return;
        } else {
            unsafe { ffmpeg::format::Input::wrap(ptr as *mut _) }
        };

        let device_name = if cfg!(target_os = "macos") {
            index.to_string()
        } else if cfg!(target_os = "linux") {
            format!("/dev/video{index}")
        } else if cfg!(target_os = "windows") {
            format!("video={index}")
        } else {
            return;
        };

        let mut options = ffmpeg::Dictionary::new();
        options.set("framerate", "30");
        if cfg!(target_os = "macos") {
            options.set("pixel_format", "uyvy422");
            options.set("video_size", "1280x720");
        }

        match ffmpeg::format::open_with(
            &std::path::PathBuf::from(&device_name),
            &ffmpeg::Format::Input(input_format),
            options,
        ) {
            Ok(ffmpeg::format::context::Context::Input(mut ictx)) => {
                let stream = ictx
                    .streams()
                    .best(ffmpeg::media::Type::Video)
                    .ok_or_else(|| anyhow::anyhow!("No video stream found"))
                    .ok();

                if let Some(stream) = stream {
                    let stream_index = stream.index();
                    let context_decoder =
                        ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                            .unwrap();
                    let mut decoder = context_decoder.decoder().video().unwrap();
                    let mut scaler = ffmpeg::software::scaling::context::Context::get(
                        decoder.format(),
                        decoder.width(),
                        decoder.height(),
                        ffmpeg::format::Pixel::RGB24,
                        decoder.width(),
                        decoder.height(),
                        ffmpeg::software::scaling::flag::Flags::BILINEAR,
                    )
                    .unwrap();

                    let mut decoded = ffmpeg::util::frame::video::Video::empty();
                    let mut cnt = 0;
                    for (stream, packet) in ictx.packets() {
                        if stream.index() != stream_index {
                            continue;
                        }

                        // Stage 1: Decode — packet → decoded frame
                        let ok = crate::perf!("DataLoader::decode-webcam::decode", {
                            decoder.send_packet(&packet).is_ok()
                                && decoder.receive_frame(&mut decoded).is_ok()
                        });
                        if !ok {
                            continue;
                        }

                        cnt += 1;
                        if (cnt - 1) % (nfv_skip + 1) != 0 {
                            continue;
                        }

                        // Stage 2: Convert — scale + row copy → Image
                        let img = crate::perf!("DataLoader::decode-webcam::convert", {
                            let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
                            scaler.run(&decoded, &mut rgb_frame).ok();

                            let width = rgb_frame.width();
                            let height = rgb_frame.height();
                            let frame_data = rgb_frame.data(0);
                            let stride = rgb_frame.stride(0);
                            let mut data = Vec::with_capacity((width * height * 3) as usize);
                            for y in 0..height as usize {
                                let row =
                                    &frame_data[y * stride..y * stride + (width as usize) * 3];
                                data.extend_from_slice(row);
                            }
                            image::RgbImage::from_raw(width, height, data).map(|rgb8| {
                                let mut img = Image::from(rgb8);
                                img.source = Some(format!("Webcam {index}").into());
                                img
                            })
                        });

                        if let Some(img) = img {
                            images.push(img);
                            if images.len() >= batch_size {
                                let to_send =
                                    std::mem::replace(images, Vec::with_capacity(batch_size));
                                if sender.send(to_send).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                tracing::error!("Failed to open webcam: {}", device_name);
            }
        }
    }

    #[cfg(feature = "video")]
    fn decode_video(
        decoder: &mut video_rs::decode::Decoder,
        nfv_skip: u64,
        batch_size: usize,
        sender: &mpsc::SyncSender<Vec<Image>>,
        images: &mut Vec<Image>,
        source: Option<std::path::PathBuf>,
    ) {
        let (w, h) = decoder.size();
        let mut cnt = 0;
        let mut iter = decoder.decode_iter();
        loop {
            // Stage 1: Decode — codec decode (expensive I/O + codec work)
            let frame = crate::perf!("DataLoader::decode-video/stream::read", iter.next());
            let (ts, frame) = match frame {
                Some(Ok(x)) => x,
                Some(Err(_)) | None => break,
            };

            cnt += 1;
            if (cnt - 1) % (nfv_skip + 1) != 0 {
                continue;
            }

            // Stage 2: Convert — buffer reinterpretation → Image
            let img = crate::perf!("DataLoader::decode-video/stream::convert", {
                image::ImageBuffer::from_raw(w as _, h as _, frame.into_raw_vec_and_offset().0).map(
                    |rgb8: image::RgbImage| {
                        let mut img = Image::from(rgb8).with_timestamp(ts.as_secs_f64());
                        img.source = source.clone();
                        img
                    },
                )
            });

            if let Some(img) = img {
                images.push(img);
                if images.len() >= batch_size {
                    let to_send = std::mem::replace(images, Vec::with_capacity(batch_size));
                    if sender.send(to_send).is_err() {
                        return;
                    }
                }
            }
        }
    }

    pub fn iter(&self) -> DataLoaderIter<'_> {
        DataLoaderIter {
            receiver: &self.receiver,
            progress_bar: self.progress_bar.as_ref(),
        }
    }

    pub fn tasks(&self) -> &VecDeque<SourceType> {
        self.source.tasks()
    }

    pub fn with_bound(mut self, x: usize) -> Self {
        self.bound = Some(x);
        self
    }

    pub fn with_batch(mut self, x: usize) -> Self {
        self.batch_size = x;
        self
    }

    pub fn nf(&self) -> u64 {
        self.nf
    }

    #[cfg(feature = "video")]
    pub fn with_nfv_skip(mut self, x: u64) -> Self {
        self.nfv_skip = x;
        self
    }

    #[cfg(feature = "video")]
    pub fn nfv_skip(&self) -> u64 {
        self.nfv_skip
    }

    pub fn with_progress_bar(mut self, x: bool) -> Self {
        self.with_progress_bar = x;
        self
    }
}
