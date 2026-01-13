use anyhow::Result;
use fast_image_resize::{
    images::{CroppedImageMut, Image as FImage},
    pixels::PixelType,
};
use image::DynamicImage;

use crate::{Image, ImageTransformInfo, ResizeAlg, ResizeMode};

impl Image {
    pub fn resize_with_info(
        &self,
        tw: u32,
        th: u32,
        resize_alg: ResizeAlg,
        mode: &ResizeMode,
        padding_value: u8,
    ) -> Result<(Self, ImageTransformInfo)> {
        if tw + th == 0 {
            anyhow::bail!("Invalid target height: {th} or width: {tw}.");
        }

        let (w0, h0) = self.dimensions();
        let mut trans_info = ImageTransformInfo::default()
            .with_width_src(w0)
            .with_height_src(h0)
            .with_width_dst(tw)
            .with_height_dst(th);

        if (w0, h0) == (tw, th) {
            return Ok((
                self.clone(),
                trans_info.with_width_scale(1.).with_height_scale(1.),
            ));
        }

        let (mut resizer, config) = resize_alg.build_fir_resizer_and_options()?;
        let x: DynamicImage = self.to_dyn();

        if let ResizeMode::FitExact { .. } = mode {
            let mut dst = FImage::new(tw, th, PixelType::U8x3);
            resizer.resize(&x, &mut dst, &config)?;
            trans_info = trans_info
                .with_height_scale(th as f32 / h0 as f32)
                .with_width_scale(tw as f32 / w0 as f32);

            Ok((Self::from_u8s(&dst.into_vec(), tw, th)?, trans_info))
        } else {
            let (w, h) = match mode {
                ResizeMode::Letterbox { .. } | ResizeMode::FitAdaptive { .. } => {
                    let r = (tw as f32 / w0 as f32).min(th as f32 / h0 as f32);
                    trans_info = trans_info.with_height_scale(r).with_width_scale(r);

                    (
                        (w0 as f32 * r).round() as u32,
                        (h0 as f32 * r).round() as u32,
                    )
                }
                ResizeMode::FitHeight { .. } => {
                    let r = th as f32 / h0 as f32;
                    trans_info = trans_info.with_height_scale(1.).with_width_scale(r);

                    ((r * w0 as f32).round() as u32, th)
                }
                ResizeMode::FitWidth { .. } => {
                    let r = tw as f32 / w0 as f32;
                    trans_info = trans_info.with_height_scale(r).with_width_scale(1.);

                    (tw, (r * h0 as f32).round() as u32)
                }
                _ => unreachable!(),
            };

            let mut dst = FImage::from_vec_u8(
                tw,
                th,
                vec![padding_value; 3 * th as usize * tw as usize],
                PixelType::U8x3,
            )?;
            let (l, t) = if let ResizeMode::Letterbox { .. } = mode {
                if w == tw {
                    (0, (th - h) / 2)
                } else {
                    ((tw - w) / 2, 0)
                }
            } else {
                (0, 0)
            };

            let mut dst_cropped = CroppedImageMut::new(&mut dst, l, t, w, h)?;
            resizer.resize(&x, &mut dst_cropped, &config)?;

            // Set padding info for letterbox mode
            trans_info = trans_info
                .with_width_pad(l as f32)
                .with_height_pad(t as f32);

            Ok((Self::from_u8s(&dst.into_vec(), tw, th)?, trans_info))
        }
    }
}
