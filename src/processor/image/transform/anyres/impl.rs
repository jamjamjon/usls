use anyhow::Result;

use crate::{Image, ImageTransformInfo};

#[derive(Debug, Clone, PartialEq)]
pub enum AnyResStrategy {
    SmolVLM {
        patch_width: u32,
        patch_height: u32,
        include_global: bool,
    },
    Moondream2 {
        patch_size: u32,
        templates: Vec<(u32, u32)>,
    },
    Grid {
        num_rows: u32,
        num_cols: u32,
        overlap: u32,
    },
}

impl Image {
    pub fn dynres_smolvlm(
        &self,
        patch_width: u32,
        patch_height: u32,
        include_global: bool,
    ) -> Result<(Vec<Self>, ImageTransformInfo)> {
        use image::GenericImageView;
        let mut patches = vec![];
        let image_rgb8 = self.to_rgb8();
        let (image_width, image_height) = image_rgb8.dimensions();

        let (_nw, _nh) = if image_width > patch_width || image_height > patch_height {
            let nw = image_width.div_ceil(patch_width);
            let nh = image_height.div_ceil(patch_height);
            let optimal_height = image_height.div_ceil(nh);
            let optimal_width = image_width.div_ceil(nw);

            for r in 0..nh {
                for c in 0..nw {
                    let x0 = c * optimal_width;
                    let y0 = r * optimal_height;
                    let x1 = (x0 + optimal_width).min(image_width);
                    let y1 = (y0 + optimal_height).min(image_height);
                    let sub_image = image_rgb8.view(x0, y0, x1 - x0, y1 - y0).to_image();
                    patches.push(Image::from(sub_image));
                }
            }
            (nw, nh)
        } else {
            (1, 1)
        };

        if include_global {
            patches.push(self.clone());
        }

        let info = ImageTransformInfo::default()
            .with_width_src(image_width)
            .with_height_src(image_height);

        Ok((patches, info))
    }

    pub fn dynres_moondream2(
        &self,
        patch_size: u32,
        templates: &[(u32, u32)],
    ) -> Result<(Vec<Self>, ImageTransformInfo)> {
        use image::GenericImageView;
        let mut patches = vec![self.clone()];
        let image = self.to_rgb8();
        let (im_width, im_height) = image.dimensions();
        let max_dim = im_width.max(im_height);

        let selected_template = if max_dim < (patch_size as f32 * 1.4) as u32 {
            (1, 1)
        } else {
            let aspect_ratio = im_width as f32 / im_height as f32;
            templates
                .iter()
                .min_by(|a, b| {
                    let diff_a = ((a.1 as f32 / a.0 as f32) - aspect_ratio).abs();
                    let diff_b = ((b.1 as f32 / b.0 as f32) - aspect_ratio).abs();
                    diff_a.partial_cmp(&diff_b).unwrap()
                })
                .copied()
                .unwrap_or((1, 1))
        };

        let patch_width = im_width / selected_template.1;
        let patch_height = im_height / selected_template.0;

        for row in 0..selected_template.0 {
            for col in 0..selected_template.1 {
                let x_min = col * patch_width;
                let y_min = row * patch_height;
                let cropped = image
                    .view(x_min, y_min, patch_width, patch_height)
                    .to_image();
                patches.push(Image::from(cropped));
            }
        }

        let info = ImageTransformInfo::default()
            .with_width_src(im_width)
            .with_height_src(im_height);

        Ok((patches, info))
    }
}
