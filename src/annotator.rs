use anyhow::Result;
use image::{ImageBuffer, RgbImage};

use crate::{auto_load, string_now, Results, CHECK_MARK, CROSS_MARK};

#[derive(Debug)]
pub struct Annotator {
    font: rusttype::Font<'static>,
    skeletons: Option<Vec<(usize, usize)>>,
    hide_conf: bool,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            font: Self::load_font(None).unwrap(),
            skeletons: None,
            hide_conf: false,
        }
    }
}

impl Annotator {
    pub fn with_skeletons(mut self, skeletons: &[(usize, usize)]) -> Self {
        self.skeletons = Some(skeletons.to_vec());
        self
    }

    pub fn with_font(mut self, path: &str) -> Self {
        self.font = Self::load_font(Some(path)).unwrap();
        self
    }

    pub fn save(&self, image: &RgbImage, saveout: &str) {
        let mut saveout = std::path::PathBuf::from("runs").join(saveout);
        if !saveout.exists() {
            std::fs::create_dir_all(&saveout).unwrap();
        }
        saveout.push(string_now("-"));
        let saveout = format!("{}.jpg", saveout.to_str().unwrap());
        match image.save(&saveout) {
            Err(err) => println!("{} Saving failed: {:?}", CROSS_MARK, err),
            Ok(_) => println!("{} Annotated image saved at: {}", CHECK_MARK, saveout),
        }
    }

    fn load_font(path: Option<&str>) -> Result<rusttype::Font<'static>> {
        let path_font = match path {
            None => auto_load("Arial.ttf")?,
            Some(p) => p.into(),
        };
        let buffer = std::fs::read(path_font)?;
        Ok(rusttype::Font::try_from_vec(buffer).unwrap())
    }

    pub fn get_color(&self, n: usize) -> (u8, u8, u8) {
        Self::color_palette()[n % Self::color_palette().len()]
    }

    pub fn plot(&self, img: &mut RgbImage, y: &Results) {
        // masks and polygons
        if let Some(masks) = y.masks() {
            for mask in masks.iter() {
                let mask_nd: ImageBuffer<image::Luma<_>, Vec<u8>> =
                    ImageBuffer::from_vec(img.width(), img.height(), mask.to_vec())
                        .expect("can not crate image from ndarray");
                // masks
                for _x in 0..img.width() {
                    for _y in 0..img.height() {
                        let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_nd, _x, _y);
                        if mask_p.0[0] > 0 {
                            let mut img_p = imageproc::drawing::Canvas::get_pixel(img, _x, _y);
                            img_p.0[0] /= 2;
                            img_p.0[1] = 255 - (255 - img_p.0[1]) / 3;
                            img_p.0[2] /= 2;
                            imageproc::drawing::Canvas::draw_pixel(img, _x, _y, img_p)
                        }
                    }
                }
                // contours
                let contours: Vec<imageproc::contours::Contour<i32>> =
                    imageproc::contours::find_contours(&mask_nd);
                for contour in contours.iter() {
                    for point in contour.points.iter() {
                        imageproc::drawing::draw_filled_circle_mut(
                            img,
                            (point.x, point.y),
                            1,
                            image::Rgb([255, 255, 255]),
                        );
                    }
                }
            }
        }

        // probs
        if let Some(probs) = y.probs() {
            let topk = 5usize;
            let (x, mut y) = (img.width() as i32 / 20, img.height() as i32 / 20);
            for k in probs.topk(topk).iter() {
                let legend = format!("{}: {:.2}", k.2.as_ref().unwrap_or(&k.0.to_string()), k.1);
                let scale = img.width().max(img.height()) as f32 / 30.0;
                let scale = rusttype::Scale::uniform(scale);
                let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend);
                y += text_h;
                imageproc::drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(x, y).of_size(text_w as u32, text_h as u32),
                    image::Rgb(self.get_color(k.0).into()),
                );
                imageproc::drawing::draw_text_mut(
                    img,
                    image::Rgb((0, 0, 0).into()),
                    x,
                    y,
                    scale,
                    &self.font,
                    &legend,
                );
            }
        }

        // bboxes
        if let Some(bboxes) = y.bboxes() {
            for bbox in bboxes.iter() {
                imageproc::drawing::draw_hollow_rect_mut(
                    img,
                    imageproc::rect::Rect::at(bbox.xmin() as i32, bbox.ymin() as i32)
                        .of_size(bbox.width() as u32, bbox.height() as u32),
                    image::Rgb(self.get_color(bbox.id()).into()),
                );
                let legend = if self.hide_conf {
                    bbox.name().unwrap_or(&bbox.id().to_string()).to_string()
                } else {
                    format!(
                        "{}: {:.4}",
                        bbox.name().unwrap_or(&bbox.id().to_string()),
                        bbox.confidence()
                    )
                };
                let scale = img.width().max(img.height()) as f32 / 45.0;
                let scale = rusttype::Scale::uniform(scale);
                let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend);
                let text_y = if bbox.ymin() as i32 > text_h {
                    bbox.ymin() as i32 - text_h
                } else {
                    text_h - bbox.ymin() as i32
                };
                imageproc::drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(bbox.xmin() as i32, text_y)
                        .of_size(text_w as u32, text_h as u32),
                    image::Rgb(self.get_color(bbox.id()).into()),
                );
                imageproc::drawing::draw_text_mut(
                    img,
                    image::Rgb((0, 0, 0).into()),
                    bbox.xmin() as i32,
                    text_y,
                    scale,
                    &self.font,
                    &legend,
                );
            }
        }

        // keypoints
        if let Some(keypoints) = y.keypoints() {
            let radius = 3;
            for kpts in keypoints.iter() {
                for (i, kpt) in kpts.iter().enumerate() {
                    if kpt.confidence() == 0.0 {
                        continue;
                    }
                    // draw point
                    imageproc::drawing::draw_filled_circle_mut(
                        img,
                        (kpt.x() as i32, kpt.y() as i32),
                        radius,
                        image::Rgb(self.get_color(i + 10).into()),
                    );
                }

                // draw skeleton
                if let Some(skeletons) = &self.skeletons {
                    for &(i, ii) in skeletons.iter() {
                        let kpt1 = &kpts[i];
                        let kpt2 = &kpts[ii];
                        if kpt1.confidence() == 0.0 || kpt2.confidence() == 0.0 {
                            continue;
                        }
                        imageproc::drawing::draw_line_segment_mut(
                            img,
                            (kpt1.x(), kpt1.y()),
                            (kpt2.x(), kpt2.y()),
                            image::Rgb([255, 51, 255]),
                        );
                    }
                }
            }
        }
    }

    fn color_palette() -> Vec<(u8, u8, u8)> {
        vec![
            (0, 255, 0),
            (255, 128, 0),
            (0, 0, 255),
            (255, 153, 51),
            (255, 0, 0),
            (255, 51, 255),
            (102, 178, 255),
            (51, 153, 255),
            (255, 51, 51),
            (153, 255, 153),
            (102, 255, 102),
            (153, 204, 255),
            (255, 153, 153),
            (255, 178, 102),
            (230, 230, 0),
            (255, 153, 255),
            (255, 102, 255),
            (255, 102, 102),
            (51, 255, 51),
            (255, 255, 255),
        ]
    }
}
