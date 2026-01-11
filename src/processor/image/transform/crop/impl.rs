use anyhow::Result;

use crate::{Image, ImageTransformInfo};

/// Crop mode variants
///
/// Reference: <https://pytorch.org/vision/master/generated/torchvision.transforms.functional.center_crop.html>
#[derive(Debug, Clone)]
pub enum CropMode {
    /// Center crop to specific size.
    ///
    /// If crop size is larger than image, crop what's available.
    /// This matches PyTorch's center_crop behavior.
    Center { width: u32, height: u32 },
    Fixed {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

impl Image {
    /// Center crop the image to the specified size
    pub fn crop_center(&self, width: u32, height: u32) -> Result<(Self, ImageTransformInfo)> {
        let (w0, h0) = self.dimensions();

        // Match torchvision.transforms.functional.center_crop:
        // - If requested crop is larger than input, pad with 0 then crop.
        // - Crop location uses round((H - crop_h)/2.0) and round((W - crop_w)/2.0)
        let (pad_left, pad_top, crop_left, crop_top) = {
            let (pad_left, crop_left) = if width > w0 {
                // padding_left = (crop_w - in_w) // 2, crop_left = 0
                (((width - w0) / 2), 0)
            } else {
                // crop_left = round((in_w - crop_w) / 2.0) == (diff + 1) // 2
                (0, (w0 - width).div_ceil(2))
            };

            let (pad_top, crop_top) = if height > h0 {
                (((height - h0) / 2), 0)
            } else {
                (0, (h0 - height).div_ceil(2))
            };

            (pad_left, pad_top, crop_left, crop_top)
        };

        let mut cropped = image::RgbImage::new(width, height);
        let src = self.image.as_raw();
        let dst = cropped.as_mut();

        // output(x, y) samples input at (x + crop_left - pad_left, y + crop_top - pad_top)
        // If out of bounds, leave 0.
        for y in 0..height {
            for x in 0..width {
                let src_x = x as i64 + crop_left as i64 - pad_left as i64;
                let src_y = y as i64 + crop_top as i64 - pad_top as i64;

                let out_idx = ((y * width + x) * 3) as usize;
                if src_x >= 0 && src_x < w0 as i64 && src_y >= 0 && src_y < h0 as i64 {
                    let in_idx = (((src_y as u32) * w0 + (src_x as u32)) * 3) as usize;
                    dst[out_idx..out_idx + 3].copy_from_slice(&src[in_idx..in_idx + 3]);
                } else {
                    dst[out_idx] = 0;
                    dst[out_idx + 1] = 0;
                    dst[out_idx + 2] = 0;
                }
            }
        }

        let info = ImageTransformInfo::default()
            .with_width_src(w0)
            .with_height_src(h0)
            .with_width_dst(width)
            .with_height_dst(height)
            .with_width_scale(1.0)
            .with_height_scale(1.0)
            .with_width_pad(pad_left as f32)
            .with_height_pad(pad_top as f32);

        Ok((Self::from(cropped), info))
    }
}
