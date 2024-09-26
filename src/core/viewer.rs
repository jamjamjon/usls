use anyhow::Result;
use image::DynamicImage;
use minifb::{Window, WindowOptions};
use video_rs::{
    encode::{Encoder, Settings},
    time::Time,
};

use crate::{string_now, Dir, Key};

pub struct Viewer<'a> {
    name: &'a str,
    window: Option<Window>,
    window_scale: f32,
    window_resizable: bool,
    fps_poll: usize,
    fps: usize,
    writer: Option<Encoder>,
    position: Time,
}

impl Default for Viewer<'_> {
    fn default() -> Self {
        Self {
            name: "usls-viewer",
            window: None,
            window_scale: 0.5,
            window_resizable: true,
            fps_poll: 100,
            fps: 25,
            writer: None,
            position: Time::zero(),
        }
    }
}

impl Viewer<'_> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn imshow(&mut self, xs: &[DynamicImage]) -> Result<()> {
        for x in xs.iter() {
            let rgb = x.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let (w_scale, h_scale) = (
                (w as f32 * self.window_scale) as usize,
                (h as f32 * self.window_scale) as usize,
            );

            // should reload?
            let should_reload = match &self.window {
                None => true,
                Some(window) => {
                    if self.window_resizable {
                        false
                    } else {
                        window.get_size() != (w_scale, h_scale)
                    }
                }
            };

            // create window
            if should_reload {
                self.window = Window::new(
                    self.name,
                    w_scale,
                    h_scale,
                    WindowOptions {
                        resize: true,
                        topmost: true,
                        borderless: false,
                        scale: minifb::Scale::X1,
                        ..WindowOptions::default()
                    },
                )
                .ok()
                .map(|mut x| {
                    x.set_target_fps(self.fps_poll);
                    x
                });
            }

            // build buffer
            let mut buffer: Vec<u32> = Vec::with_capacity(w * h);
            for pixel in rgb.pixels() {
                let r = pixel[0];
                let g = pixel[1];
                let b = pixel[2];
                let p = Self::rgb8_to_u32(r, g, b);
                buffer.push(p);
            }

            // update buffer
            self.window
                .as_mut()
                .unwrap()
                .update_with_buffer(&buffer, w, h)?;
        }

        Ok(())
    }

    pub fn write(&mut self, frame: &image::DynamicImage) -> Result<()> {
        // build writer at the 1st time
        let frame = frame.to_rgb8();
        let (w, h) = frame.dimensions();
        if self.writer.is_none() {
            let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
            let saveout = Dir::saveout(&["runs"])?.join(format!("{}.mp4", string_now("-")));
            tracing::info!("Video will be save to: {:?}", saveout);
            self.writer = Some(Encoder::new(saveout, settings)?);
        }

        // write video
        if let Some(writer) = self.writer.as_mut() {
            let raw_data = frame.to_vec();
            let frame = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), raw_data)?;

            // encode and update
            writer.encode(&frame, self.position)?;
            self.position = self
                .position
                .aligned_with(Time::from_nth_of_a_second(self.fps))
                .add();
        }
        Ok(())
    }

    pub fn write_batch(&mut self, frames: &[image::DynamicImage]) -> Result<()> {
        for frame in frames.iter() {
            self.write(frame)?
        }
        Ok(())
    }

    pub fn finish_write(&mut self) -> Result<()> {
        match &mut self.writer {
            Some(writer) => writer.finish()?,
            None => {
                tracing::info!("Found no video writer. No need to release.");
            }
        }
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        if let Some(window) = &self.window {
            window.is_open()
        } else {
            false
        }
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        if let Some(window) = &self.window {
            window.is_key_down(key)
        } else {
            false
        }
    }

    pub fn is_esc_pressed(&self) -> bool {
        self.is_key_pressed(Key::Escape)
    }

    pub fn resizable(mut self, x: bool) -> Self {
        self.window_resizable = x;
        self
    }

    pub fn with_scale(mut self, x: f32) -> Self {
        self.window_scale = x;
        self
    }

    pub fn with_fps(mut self, x: usize) -> Self {
        self.fps = x;
        self
    }

    pub fn with_delay(mut self, x: usize) -> Self {
        self.fps_poll = 1000 / x;
        self
    }

    pub fn wh(&self) -> Option<(usize, usize)> {
        self.window.as_ref().map(|x| x.get_size())
    }

    fn rgb8_to_u32(r: u8, g: u8, b: u8) -> u32 {
        let (r, g, b) = (r as u32, g as u32, b as u32);
        (r << 16) | (g << 8) | b
    }
}
