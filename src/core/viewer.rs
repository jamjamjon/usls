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
    fps_poll: usize,
    writer: Option<Encoder>,
    position: Time,
    saveout: Option<String>,
    saveout_base: String,
    saveout_subs: Vec<String>,
}

impl Default for Viewer<'_> {
    fn default() -> Self {
        Self {
            name: "usls-viewer",
            window: None,
            fps_poll: 100,
            writer: None,
            position: Time::zero(),
            saveout: None,
            saveout_subs: vec![],
            saveout_base: String::from("runs"),
        }
    }
}

impl Viewer<'_> {
    pub fn new() -> Self {
        Default::default()
    }

    /// Create folders for saving annotated results. e.g., `./runs/xxx`
    pub fn saveout(&self) -> Result<std::path::PathBuf> {
        let mut subs = vec![self.saveout_base.as_str()];
        if let Some(saveout) = &self.saveout {
            // add subs
            if !self.saveout_subs.is_empty() {
                let xs = self
                    .saveout_subs
                    .iter()
                    .map(|x| x.as_str())
                    .collect::<Vec<&str>>();
                subs.extend(xs);
            }

            // add filename
            subs.push(saveout);
        }

        // mkdir even no filename specified
        Dir::Currnet.raw_path_with_subs(&subs)
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

    pub fn wh(&self) -> Option<(usize, usize)> {
        self.window.as_ref().map(|x| x.get_size())
    }

    pub fn imshow(&mut self, xs: &[DynamicImage]) -> Result<()> {
        for x in xs.iter() {
            let rgb = x.to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);

            // check if need to reload
            let reload_window = match &self.window {
                Some(window) => window.get_size() != (w, h),
                None => true,
            };

            // reload
            if reload_window {
                self.window = Window::new(
                    self.name,
                    w as _,
                    h as _,
                    WindowOptions {
                        resize: true,
                        topmost: true,
                        borderless: false,
                        ..WindowOptions::default()
                    },
                )
                .ok()
                .map(|mut x| {
                    x.set_target_fps(self.fps_poll);
                    x
                });
            }

            // update buffer
            let mut buffer: Vec<u32> = Vec::with_capacity(w * h);
            for pixel in rgb.pixels() {
                let r = pixel[0];
                let g = pixel[1];
                let b = pixel[2];
                let p = Self::from_u8_rgb(r, g, b);
                buffer.push(p);
            }

            // Update the window with the image buffer
            self.window
                .as_mut()
                .unwrap()
                .update_with_buffer(&buffer, w, h)?;

            // optional: save as videos
        }

        Ok(())
    }

    pub fn write(&mut self, frame: &image::DynamicImage, fps: usize) -> Result<()> {
        // build writer at the 1st time
        let frame = frame.to_rgb8();
        let (w, h) = frame.dimensions();
        if self.writer.is_none() {
            let settings = Settings::preset_h264_yuv420p(w as _, h as _, false);
            let saveout = self.saveout()?.join(format!("{}.mp4", string_now("-")));
            self.writer = Some(Encoder::new(saveout, settings)?);
        }

        // write video
        if let Some(writer) = self.writer.as_mut() {
            // let raw_data = frame.into_raw();
            let raw_data = frame.to_vec();
            let frame = ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), raw_data)?;

            // encode and update
            writer.encode(&frame, self.position)?;
            self.position = self
                .position
                .aligned_with(Time::from_nth_of_a_second(fps))
                .add();
        }
        Ok(())
    }

    pub fn write_batch(&mut self, frames: &[image::DynamicImage], fps: usize) -> Result<()> {
        for frame in frames.iter() {
            self.write(frame, fps)?
        }
        Ok(())
    }

    pub fn finish_write(&mut self) -> Result<()> {
        match &mut self.writer {
            Some(writer) => Ok(writer.finish()?),
            None => anyhow::bail!("Found no video encoder."),
        }
    }

    fn from_u8_rgb(r: u8, g: u8, b: u8) -> u32 {
        let (r, g, b) = (r as u32, g as u32, b as u32);
        (r << 16) | (g << 8) | b
    }
}
