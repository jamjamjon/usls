//! Viewer module for high-performance real-time visualization

use anyhow::{Context, Result};
use minifb::{Key, ScaleMode, Window, WindowOptions};
#[cfg(feature = "video")]
use video_rs::{
    encode::{Encoder, Settings},
    frame::Frame,
    time::Time,
};

use crate::Image;

/// High-performance real-time visualization with an OpenCV-like `imshow` API.
///
/// # 📺 Viewer
///
/// A lightweight, high-speed image viewer designed for computer vision and machine learning
/// applications. Provides an OpenCV-like interface for displaying images with minimal overhead.
///
/// ## Features
///
/// - **Display**: High-speed `imshow` with automatic window management and scaling
/// - **Interactivity**: Blocking or non-blocking key event handling via `wait_key`
/// - **State**: Monitor window status (`is_open`, `is_closed`) and key events (`is_esc_pressed`, `is_key_pressed`)
/// - **Recording**: Save displayed frames to a video file (requires `video` feature)
///
/// ## Examples
///
/// ### Basic Image Display
///
/// ```no_run
/// use usls::{Viewer, Image};
///
/// let mut viewer = Viewer::new("Image Display")
///     .with_window_scale(1.5); // Scale window by 1.5x
///
/// // Display an image
/// viewer.imshow(&image)?;
///
/// // Wait for key press (non-blocking)
/// if let Some(key) = viewer.wait_key(1000) {
///     println!("Key pressed: {:?}", key);
/// }
/// ```
///
/// ### Video/Image Stream Processing
///
/// ```no_run
/// use usls::{Viewer, DataLoader};
///
/// let dl = DataLoader::new("./video.mp4")?.stream()?;
/// let mut viewer = Viewer::default().with_window_scale(1.2);
///
/// for images in &dl {
///     // Check window status - exit if user closed window
///     if viewer.is_window_exist_and_closed() {
///         break;
///     }
///
///     // Display first image from batch
///     viewer.imshow(&images[0])?;
///
///     // Handle key events with delay
///     if let Some(key) = viewer.wait_key(30) {
///         if key == usls::Key::Escape {
///             break; // Exit on ESC
///         }
///     }
/// }
/// ```
///
/// ## Video Recording
///
/// ```no_run
/// #[cfg(feature = "video")]
/// # {
/// use usls::{Viewer, DataLoader};
///
/// let dl = DataLoader::new("./video.mp4")?.stream()?;
/// let mut viewer = Viewer::default().with_window_scale(1.2);
///
/// for images in &dl {
///     // Exit conditions
///     if viewer.is_window_exist_and_closed() {
///         break;
///     }
///
///     // Display frame
///     viewer.imshow(&images[0])?;
///     
///     // Record to video file
///     viewer.write_video_frame(&images[0])?;
///
///     // Handle key events
///     if let Some(key) = viewer.wait_key(30) {
///         if key == usls::Key::Escape {
///             break;
///         }
///     }
/// }
///
/// // Video is automatically finalized when viewer is dropped
/// # }
/// ```
///
/// # Performance
///
/// - Uses `minifb` for minimal overhead windowing
/// - Automatic buffer management and window resizing
/// - Efficient key event handling with pending key queue
/// - Zero-copy image display when possible
///
/// # Feature Flags
///
/// - `viewer`: Enable visualization functionality (required)
/// - `video`: Enable video recording capabilities
///
/// > ⚠️ Visualization requires the `viewer` feature.
///
/// See Example: [`dataloader_viewer/imshow.rs`](../../examples/dataloader_viewer/imshow.rs)
pub struct Viewer<'a> {
    window: Option<Window>,
    window_title: &'a str,
    window_scale: f32,
    buffer: Vec<u32>,
    image_height: usize,
    image_width: usize,
    pending_keys: Vec<Key>, // Store keys captured during imshow's update to prevent them from being swallowed
    #[cfg(feature = "video")]
    video_encoders: std::collections::HashMap<std::path::PathBuf, Encoder>,
}

impl Default for Viewer<'_> {
    fn default() -> Self {
        Self {
            window: None,
            window_title: "Untitled",
            window_scale: 1.,
            buffer: vec![],
            image_height: 0,
            image_width: 0,
            #[cfg(feature = "video")]
            video_encoders: std::collections::HashMap::new(),
            pending_keys: vec![],
        }
    }
}

impl Viewer<'_> {
    /// Creates a new Viewer with the specified window title.
    pub fn new(window_title: &str) -> Viewer<'_> {
        Viewer {
            window: None,
            window_title,
            window_scale: 1.0,
            buffer: Vec::new(),
            image_height: 0,
            image_width: 0,
            #[cfg(feature = "video")]
            video_encoders: std::collections::HashMap::new(),
            pending_keys: Vec::new(),
        }
    }

    #[inline]
    fn calculate_window_size(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        (
            (img_w as f32 * self.window_scale).round() as usize,
            (img_h as f32 * self.window_scale).round() as usize,
        )
    }

    #[inline]
    fn needs_window_resize(&self, target_w: usize, target_h: usize) -> bool {
        self.window.as_ref().is_none_or(|win| {
            let (curr_w, curr_h) = win.get_size();
            curr_w != target_w || curr_h != target_h
        })
    }

    fn create_window(&mut self, width: usize, height: usize) -> Result<()> {
        self.window = None;
        let options = WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Stretch,
            ..WindowOptions::default()
        };
        let mut window = Window::new(self.window_title, width, height, options)
            .context("Failed to create window")?;
        window.set_target_fps(0);
        self.window = Some(window);
        Ok(())
    }

    pub fn imshow(&mut self, rgb: &Image) -> Result<()> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let (win_w, win_h) = self.calculate_window_size(w, h);

        if win_w == 0 || win_h == 0 {
            return Ok(());
        }

        rgb.to_u32s_into(&mut self.buffer);
        self.image_width = w;
        self.image_height = h;

        if self.needs_window_resize(win_w, win_h) {
            self.create_window(win_w, win_h)?;
        }

        if let Some(window) = self.window.as_mut() {
            window
                .update_with_buffer(&self.buffer, w, h)
                .context("Failed to update window buffer")?;

            let keys = window.get_keys_pressed(minifb::KeyRepeat::No);
            if !keys.is_empty() {
                self.pending_keys.extend(keys);
            }
        }

        Ok(())
    }

    pub fn wait_key(&mut self, delay_ms: u64) -> Option<Key> {
        if !self.pending_keys.is_empty() {
            return Some(self.pending_keys.remove(0));
        }

        let window = self.window.as_mut()?;

        if delay_ms == 0 {
            window.update();
            return window
                .get_keys_pressed(minifb::KeyRepeat::No)
                .first()
                .copied();
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(delay_ms);
        let img_w = self.image_width;
        let img_h = self.image_height;
        let (win_w, win_h) = (
            (img_w as f32 * self.window_scale).round() as usize,
            (img_h as f32 * self.window_scale).round() as usize,
        );
        let use_full_refresh = delay_ms > 30;

        loop {
            if std::time::Instant::now() >= deadline {
                break;
            }

            let size_changed = window.get_size() != (win_w, win_h);
            if size_changed || use_full_refresh {
                if let Err(e) = window.update_with_buffer(&self.buffer, img_w, img_h) {
                    tracing::warn!("Failed to refresh window buffer: {}", e);
                    window.update();
                }
            } else {
                window.update();
            }

            if let Some(key) = window
                .get_keys_pressed(minifb::KeyRepeat::No)
                .first()
                .copied()
            {
                return Some(key);
            }

            if !window.is_open() {
                break;
            }

            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        None
    }

    #[cfg(feature = "video")]
    pub fn write_video_frame(&mut self, frame: &Image) -> Result<()> {
        if let Some(ts) = frame.timestamp {
            let (w, h) = frame.dimensions();
            let source_path = frame
                .source
                .clone()
                .unwrap_or_else(|| std::path::PathBuf::from("unknown_source"));

            if !self.video_encoders.contains_key(&source_path) {
                // Aggressively finalize other encoders if we switch to a new source.
                // This is crucial for memory management when processing many videos sequentially.
                if !self.video_encoders.is_empty() {
                    tracing::info!(
                        "New video source detected: {:?}. Finalizing previous encoders to release memory.",
                        source_path
                    );
                    self.finalize_video()?;
                }

                let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
                // Use default saving logic based on timestamp and source name
                let saveout = {
                    let p = crate::Dir::Current
                        .base_dir_with_subs(&["runs"])?
                        .join(format!("{}.mov", crate::timestamp(Some("-"))));

                    let parent = p.parent().unwrap_or_else(|| std::path::Path::new("."));
                    let src_stem = source_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("output");

                    parent.join(format!("{}_{}.mov", src_stem, crate::timestamp(Some("-"))))
                };

                tracing::info!(
                    "Video source {:?} will be saved to: {:?}",
                    source_path,
                    saveout
                );
                self.video_encoders
                    .insert(source_path.clone(), Encoder::new(saveout, settings)?);
            }

            if let Some(encoder) = self.video_encoders.get_mut(&source_path) {
                encoder.encode(
                    &Frame::from_shape_vec((h as usize, w as usize, 3), frame.as_raw().clone())?,
                    Time::from_secs_f64(ts),
                )?;
            }
        } else {
            tracing::warn!("Frame timestamp is not available, failed to encode video.");
        }

        Ok(())
    }

    #[cfg(feature = "video")]
    pub fn finalize_video(&mut self) -> Result<()> {
        let encoders = std::mem::take(&mut self.video_encoders);
        for (source, mut encoder) in encoders {
            match encoder.finish() {
                Ok(_) => tracing::debug!("Video encoding for {:?} finalized successfully.", source),
                Err(err) => {
                    tracing::warn!("Error finalizing video encoding for {:?}: {}", source, err)
                }
            }
        }
        Ok(())
    }

    pub fn is_window_open(&self) -> bool {
        self.window.as_ref().is_some_and(|w| w.is_open())
    }

    pub fn is_window_exist(&self) -> bool {
        self.window.is_some()
    }

    pub fn is_window_exist_and_open(&self) -> bool {
        self.is_window_exist() && self.is_window_open()
    }

    pub fn is_window_exist_and_closed(&self) -> bool {
        self.is_window_exist() && !self.is_window_open()
    }

    pub fn is_key_down(&self, key: Key) -> bool {
        self.window.as_ref().is_some_and(|w| w.is_key_down(key))
    }

    pub fn is_key_pressed(&mut self, key: Key) -> bool {
        self.window
            .as_mut()
            .is_some_and(|w| w.is_key_pressed(key, minifb::KeyRepeat::No))
    }

    pub fn is_key_released(&self, key: Key) -> bool {
        self.window.as_ref().is_some_and(|w| w.is_key_released(key))
    }

    pub fn is_esc_pressed(&mut self) -> bool {
        self.is_key_pressed(Key::Escape)
    }

    pub fn get_the_1st_key_pressed(&self) -> Option<Key> {
        self.window
            .as_ref()
            .and_then(|w| w.get_keys_pressed(minifb::KeyRepeat::No).first().copied())
    }

    pub fn get_the_1st_key_released(&self) -> Option<Key> {
        self.window
            .as_ref()
            .and_then(|w| w.get_keys_released().first().copied())
    }

    pub fn with_window_scale(mut self, x: f32) -> Self {
        self.window_scale = x;
        self
    }
}

impl Drop for Viewer<'_> {
    fn drop(&mut self) {
        #[cfg(feature = "video")]
        let _ = self.finalize_video();
    }
}
