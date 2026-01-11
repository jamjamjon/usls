use anyhow::Result;
use image::RgbImage;

use crate::{Image, ImageTransformInfo};

#[derive(Debug, Clone)]
pub enum PadMode {
    ToMultiple {
        window_size: usize,
        fill_mode: PadFillMode,
    },
    ToSize {
        width: u32,
        height: u32,
        fill_mode: PadFillMode,
        align: PadAlign,
    },
    Fixed {
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        fill_mode: PadFillMode,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum PadFillMode {
    Constant(u8),
    Reflect,
    Replicate,
    Wrap,
}

#[derive(Debug, Clone, Copy)]
pub enum PadAlign {
    TopLeft,
    Center,
}

impl Image {
    /// Pad the image to the nearest multiple of window_size (using Reflect mode)
    pub fn pad(&self, window_size: usize) -> Result<(Self, ImageTransformInfo)> {
        self.pad_with_mode(window_size, PadFillMode::Reflect)
    }

    /// Pad the image to the nearest multiple of window_size with specified fill mode
    pub fn pad_with_mode(
        &self,
        window_size: usize,
        fill_mode: PadFillMode,
    ) -> Result<(Self, ImageTransformInfo)> {
        let (width, height) = self.image.dimensions();
        let (w_old, h_old) = (width as usize, height as usize);

        //  (size // window_size + 1) * window_size - size
        let h_pad_total = (h_old / window_size + 1) * window_size - h_old;
        let w_pad_total = (w_old / window_size + 1) * window_size - w_old;

        // Create new image with padded dimensions
        let (new_w, new_h) = (w_old + w_pad_total, h_old + h_pad_total);
        let mut padded = RgbImage::new(new_w as u32, new_h as u32);

        match fill_mode {
            crate::PadFillMode::Constant(value) => {
                for pixel in padded.pixels_mut() {
                    pixel.0 = [value, value, value];
                }

                let src_pixels = self.image.as_raw();
                let dst_pixels = padded.as_mut();
                for y in 0..h_old {
                    let src_offset = y * w_old * 3;
                    let dst_offset = y * new_w * 3;
                    let row_bytes = w_old * 3;
                    dst_pixels[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                }
            }
            PadFillMode::Reflect => {
                let src_pixels = self.image.as_raw();
                let dst_pixels = padded.as_mut();

                for y in 0..h_old {
                    let src_offset = y * w_old * 3;
                    let dst_offset = y * new_w * 3;
                    let row_bytes = w_old * 3;
                    dst_pixels[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                }

                if h_pad_total > 0 {
                    for h_idx in 0..h_pad_total {
                        let src_h = h_old - 1 - (h_idx % h_old);
                        let src_offset = src_h * w_old * 3;
                        let dst_offset = (h_old + h_idx) * new_w * 3;
                        let row_bytes = w_old * 3;
                        dst_pixels[dst_offset..dst_offset + row_bytes]
                            .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                    }
                }

                if w_pad_total > 0 {
                    for h_idx in 0..new_h {
                        let row_offset = h_idx * new_w * 3;
                        for w_idx in 0..w_pad_total {
                            let src_w = w_old - 1 - (w_idx % w_old);
                            let src_pixel_offset = row_offset + src_w * 3;
                            let dst_pixel_offset = row_offset + (w_old + w_idx) * 3;
                            let temp_pixel = [
                                dst_pixels[src_pixel_offset],
                                dst_pixels[src_pixel_offset + 1],
                                dst_pixels[src_pixel_offset + 2],
                            ];
                            dst_pixels[dst_pixel_offset..dst_pixel_offset + 3]
                                .copy_from_slice(&temp_pixel);
                        }
                    }
                }
            }
            PadFillMode::Replicate => {
                let src_pixels = self.image.as_raw();
                let dst_pixels = padded.as_mut();

                for y in 0..h_old {
                    let src_offset = y * w_old * 3;
                    let dst_offset = y * new_w * 3;
                    let row_bytes = w_old * 3;
                    dst_pixels[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                }

                if h_pad_total > 0 {
                    let last_row_src = (h_old - 1) * w_old * 3;
                    for h_idx in 0..h_pad_total {
                        let dst_offset = (h_old + h_idx) * new_w * 3;
                        dst_pixels[dst_offset..dst_offset + w_old * 3]
                            .copy_from_slice(&src_pixels[last_row_src..last_row_src + w_old * 3]);
                    }
                }

                if w_pad_total > 0 {
                    for h_idx in 0..new_h {
                        let row_offset = h_idx * new_w * 3;
                        let last_pixel_offset = row_offset + (w_old - 1) * 3;
                        let last_pixel = [
                            dst_pixels[last_pixel_offset],
                            dst_pixels[last_pixel_offset + 1],
                            dst_pixels[last_pixel_offset + 2],
                        ];
                        for w_idx in 0..w_pad_total {
                            let dst_pixel_offset = row_offset + (w_old + w_idx) * 3;
                            dst_pixels[dst_pixel_offset..dst_pixel_offset + 3]
                                .copy_from_slice(&last_pixel);
                        }
                    }
                }
            }
            PadFillMode::Wrap => {
                let src_pixels = self.image.as_raw();
                let dst_pixels = padded.as_mut();

                for y in 0..h_old {
                    let src_offset = y * w_old * 3;
                    let dst_offset = y * new_w * 3;
                    let row_bytes = w_old * 3;
                    dst_pixels[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                }

                if h_pad_total > 0 {
                    for h_idx in 0..h_pad_total {
                        let src_h = h_idx % h_old;
                        let src_offset = src_h * w_old * 3;
                        let dst_offset = (h_old + h_idx) * new_w * 3;
                        let row_bytes = w_old * 3;
                        dst_pixels[dst_offset..dst_offset + row_bytes]
                            .copy_from_slice(&src_pixels[src_offset..src_offset + row_bytes]);
                    }
                }

                if w_pad_total > 0 {
                    for h_idx in 0..new_h {
                        let row_offset = h_idx * new_w * 3;
                        for w_idx in 0..w_pad_total {
                            let src_w = w_idx % w_old;
                            let src_pixel_offset = row_offset + src_w * 3;
                            let dst_pixel_offset = row_offset + (w_old + w_idx) * 3;
                            let temp_pixel = [
                                dst_pixels[src_pixel_offset],
                                dst_pixels[src_pixel_offset + 1],
                                dst_pixels[src_pixel_offset + 2],
                            ];
                            dst_pixels[dst_pixel_offset..dst_pixel_offset + 3]
                                .copy_from_slice(&temp_pixel);
                        }
                    }
                }
            }
        }

        let images_transform_info = ImageTransformInfo::default()
            .with_width_src(width)
            .with_height_src(height)
            .with_width_dst(new_w as u32)
            .with_height_dst(new_h as u32)
            .with_height_pad(h_pad_total as f32)
            .with_width_pad(w_pad_total as f32);

        Ok((Self::from(padded), images_transform_info))
    }
}
