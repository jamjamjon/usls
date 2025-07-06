use anyhow::{Context, Result};
use minifb::{Key, ScaleMode, Window, WindowOptions};
#[cfg(feature = "video")]
use video_rs::{
    encode::{Encoder, Settings},
    frame::Frame,
    time::Time,
};

use crate::Image;

pub struct Viewer<'a> {
    window: Option<Window>,
    window_title: &'a str,
    window_scale: f32,
    buffer: Vec<u32>,
    image_height: usize,
    image_width: usize,
    #[cfg(feature = "video")]
    fps: usize,
    #[cfg(feature = "video")]
    video_encoder: Option<Encoder>,
    #[cfg(feature = "video")]
    position: Time,
    #[cfg(feature = "video")]
    saveout: Option<String>,
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
            fps: 25,
            #[cfg(feature = "video")]
            video_encoder: None,
            #[cfg(feature = "video")]
            position: Time::zero(),
            #[cfg(feature = "video")]
            saveout: None,
        }
    }
}

impl Viewer<'_> {
    pub fn new(window_title: &str) -> Viewer<'_> {
        Viewer {
            window_title,
            ..Default::default()
        }
    }

    pub fn imshow(&mut self, rgb: &Image) -> Result<()> {
        (self.buffer, self.image_height, self.image_width) =
            (rgb.to_u32s(), rgb.height() as usize, rgb.width() as usize);

        if self.window.is_none() {
            let (w, h) = (
                (self.image_width as f32 * self.window_scale) as usize,
                (self.image_height as f32 * self.window_scale) as usize,
            );
            self.window = Some(Self::create_window(self.window_title, w, h)?);
        }

        if let Some(window) = self.window.as_mut() {
            window
                .update_with_buffer(&self.buffer, self.image_width, self.image_height)
                .context("Failed to update window buffer")?;
        } else {
            anyhow::bail!("Window is not initialized");
        }

        Ok(())
    }

    pub fn wait_key(&mut self, delay_ms: u64) -> Option<Key> {
        if let Some(window) = &mut self.window {
            let t = if delay_ms > 0 {
                Some(std::time::Instant::now() + std::time::Duration::from_millis(delay_ms))
            } else {
                None
            };

            loop {
                if window.get_size()
                    != (
                        (self.image_width as f32 * self.window_scale) as usize,
                        (self.image_height as f32 * self.window_scale) as usize,
                    )
                {
                    if let Err(e) =
                        window.update_with_buffer(&self.buffer, self.image_width, self.image_height)
                    {
                        log::error!("Failed to update buffer: {}", e);
                        // Try to continue with regular update instead of crashing
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

                if let Some(t) = t {
                    if std::time::Instant::now() >= t {
                        break;
                    }
                }

                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        None
    }

    #[cfg(feature = "video")]
    pub fn write_video_frame(&mut self, frame: &Image) -> Result<()> {
        let (w, h) = frame.dimensions();
        if self.video_encoder.is_none() {
            let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
            let saveout = match &self.saveout {
                None => crate::Dir::Current
                    .base_dir_with_subs(&["runs"])?
                    .join(format!("{}.mov", crate::timestamp(Some("-")))),
                Some(x) => std::path::PathBuf::from(x),
            };
            log::info!("Video will be save to: {:?}", saveout);
            self.video_encoder = Some(Encoder::new(saveout, settings)?);
        }

        if let Some(video_encoder) = self.video_encoder.as_mut() {
            video_encoder.encode(
                &Frame::from_shape_vec((h as usize, w as usize, 3), frame.as_raw().clone())?,
                self.position,
            )?;
            self.position = self
                .position
                .aligned_with(Time::from_nth_of_a_second(self.fps))
                .add();
        }

        Ok(())
    }

    #[cfg(feature = "video")]
    pub fn finalize_video(&mut self) -> Result<()> {
        if let Some(mut video_encoder) = self.video_encoder.take() {
            match video_encoder.finish() {
                Ok(_) => log::debug!("Video encoding finalized successfully."),
                Err(err) => anyhow::bail!("Error finalizing video encoding: {}", err),
            }
        } else {
            log::debug!("No video encoder was initialized. No need to finalize.");
        }
        Ok(())
    }

    pub fn is_window_open(&self) -> bool {
        if let Some(window) = &self.window {
            window.is_open()
        } else {
            false
        }
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
        if let Some(window) = &self.window {
            window.is_key_down(key)
        } else {
            false
        }
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        if let Some(window) = &self.window {
            window.is_key_pressed(key, minifb::KeyRepeat::No)
        } else {
            false
        }
    }

    pub fn is_key_released(&self, key: Key) -> bool {
        if let Some(window) = &self.window {
            window.is_key_released(key)
        } else {
            false
        }
    }

    pub fn is_esc_pressed(&self) -> bool {
        self.is_key_pressed(Key::Escape)
    }

    pub fn get_the_1st_key_pressed(&self) -> Option<Key> {
        if let Some(window) = &self.window {
            window
                .get_keys_pressed(minifb::KeyRepeat::No)
                .first()
                .copied()
        } else {
            None
        }
    }

    pub fn get_the_1st_key_released(&self) -> Option<Key> {
        if let Some(window) = &self.window {
            window.get_keys_released().first().copied()
        } else {
            None
        }
    }

    pub fn with_window_scale(mut self, x: f32) -> Self {
        self.window_scale = x;
        self
    }

    #[cfg(feature = "video")]
    pub fn with_fps(mut self, x: usize) -> Self {
        self.fps = x;
        self
    }

    #[cfg(feature = "video")]
    pub fn with_saveout(mut self, x: String) -> Self {
        self.saveout = Some(x);
        self
    }

    #[inline]
    pub fn width_height(&self) -> Option<(usize, usize)> {
        self.window.as_ref().map(|x| x.get_size())
    }

    fn create_window(title: &str, width: usize, height: usize) -> Result<Window> {
        Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                topmost: false,
                scale_mode: ScaleMode::AspectRatioStretch,
                ..WindowOptions::default()
            },
        )
        .map(|mut x| {
            x.set_target_fps(0);
            x
        })
        .map_err(|e| anyhow::anyhow!("{}", e))
    }
}
