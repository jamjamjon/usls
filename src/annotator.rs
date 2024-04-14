use crate::{
    auto_load, string_now, Bbox, Embedding, Keypoint, Polygon, Ys, CHECK_MARK, CROSS_MARK,
};
use ab_glyph::{FontVec, PxScale};
use anyhow::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Rgb, RgbImage};

#[derive(Debug)]
pub struct Annotator {
    font: ab_glyph::FontVec,
    scale_: f32, // Cope with ab_glyph & imageproc=0.24.0
    skeletons: Option<Vec<(usize, usize)>>,
    polygon_color: Rgb<u8>,
    saveout: Option<String>,
    without_conf: bool,
    without_name: bool,
    without_bboxes: bool,
    without_masks: bool,
    without_polygons: bool,
    without_keypoints: bool,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            font: Self::load_font(None).unwrap(),
            scale_: 6.666667,
            skeletons: None,
            polygon_color: Rgb([255, 255, 255]),
            saveout: None,
            without_conf: false,
            without_name: false,
            without_bboxes: false,
            without_masks: false,
            without_polygons: false,
            without_keypoints: false,
        }
    }
}

impl Annotator {
    pub fn without_conf(mut self, x: bool) -> Self {
        self.without_conf = x;
        self
    }

    pub fn without_name(mut self, x: bool) -> Self {
        self.without_name = x;
        self
    }

    pub fn without_bboxes(mut self, x: bool) -> Self {
        self.without_bboxes = x;
        self
    }

    pub fn without_masks(mut self, x: bool) -> Self {
        self.without_masks = x;
        self
    }

    pub fn without_polygons(mut self, x: bool) -> Self {
        self.without_polygons = x;
        self
    }
    pub fn without_keypoints(mut self, x: bool) -> Self {
        self.without_keypoints = x;
        self
    }

    pub fn with_saveout(mut self, saveout: &str) -> Self {
        self.saveout = Some(saveout.to_string());
        self
    }

    pub fn with_polygon_color(mut self, rgb: [u8; 3]) -> Self {
        self.polygon_color = Rgb(rgb);
        self
    }

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

    pub fn annotate(&self, imgs: &[DynamicImage], ys: &[Ys]) {
        for (img, y) in imgs.iter().zip(ys.iter()) {
            let mut img_rgb = img.to_rgb8();

            // masks
            if !self.without_masks {
                if let Some(masks) = &y.masks {
                    self.plot_masks(&mut img_rgb, masks)
                }
            }

            // polygons
            if !self.without_polygons {
                if let Some(polygons) = &y.polygons {
                    self.plot_polygons(&mut img_rgb, polygons)
                }
            }

            // bboxes
            if !self.without_bboxes {
                if let Some(bboxes) = &y.bboxes {
                    self.plot_bboxes(&mut img_rgb, bboxes)
                }
            }

            // keypoints
            if !self.without_keypoints {
                if let Some(keypoints) = &y.keypoints {
                    self.plot_keypoints(&mut img_rgb, keypoints)
                }
            }

            // probs
            if let Some(probs) = &y.probs {
                self.plot_probs(&mut img_rgb, probs)
            }

            if let Some(saveout) = &self.saveout {
                self.save(&img_rgb, saveout);
            }
        }
    }

    pub fn plot_masks(&self, img: &mut RgbImage, masks: &[Vec<u8>]) {
        for mask in masks.iter() {
            let mask_nd: GrayImage =
                ImageBuffer::from_vec(img.width(), img.height(), mask.to_vec())
                    .expect("can not crate image from ndarray");
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
        }
    }

    pub fn plot_bboxes(&self, img: &mut RgbImage, bboxes: &[Bbox]) {
        for bbox in bboxes.iter() {
            imageproc::drawing::draw_hollow_rect_mut(
                img,
                imageproc::rect::Rect::at(bbox.xmin().round() as i32, bbox.ymin().round() as i32)
                    .of_size(bbox.width().round() as u32, bbox.height().round() as u32),
                image::Rgb(self.get_color(bbox.id()).into()),
            );
            let mut legend = String::new();
            if !self.without_name {
                legend.push_str(&bbox.name().unwrap_or(&bbox.id().to_string()).to_string());
            }
            if !self.without_conf {
                if !self.without_name {
                    legend.push_str(&format!(": {:.4}", bbox.confidence()));
                } else {
                    legend.push_str(&format!("{:.4}", bbox.confidence()));
                }
            }
            let scale_dy = img.width().max(img.height()) as f32 / 40.0;
            let scale = PxScale::from(scale_dy);
            let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend); // u32
            let text_h = text_h + text_h / 3;
            let top = if bbox.ymin() > text_h as f32 {
                (bbox.ymin().round() as u32 - text_h) as i32
            } else {
                (text_h - bbox.ymin().round() as u32) as i32
            };

            // text
            if !legend.is_empty() {
                imageproc::drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(bbox.xmin() as i32, top).of_size(text_w, text_h),
                    image::Rgb(self.get_color(bbox.id()).into()),
                );
                imageproc::drawing::draw_text_mut(
                    img,
                    image::Rgb([0, 0, 0]),
                    bbox.xmin() as i32,
                    top - (scale_dy / self.scale_).floor() as i32 + 2,
                    scale,
                    &self.font,
                    &legend,
                );
            }
        }
    }

    pub fn plot_polygons(&self, img: &mut RgbImage, polygons: &[Polygon]) {
        for polygon in polygons.iter() {
            // option: draw polygon
            let polygon = polygon
                .points
                .iter()
                .map(|p| imageproc::point::Point::new(p.x, p.y))
                .collect::<Vec<_>>();
            imageproc::drawing::draw_hollow_polygon_mut(img, &polygon, self.polygon_color);

            // option: draw circle
            // polygon.points.iter().for_each(|point| {
            //     imageproc::drawing::draw_filled_circle_mut(
            //         img,
            //         (point.x as i32, point.y as i32),
            //         1,
            //         // image::Rgb([255, 255, 255]),
            //         self.polygon_color,
            //     );
            // });
        }
    }

    pub fn plot_probs(&self, img: &mut RgbImage, probs: &Embedding) {
        let topk = 5usize;
        let (x, mut y) = (img.width() as i32 / 20, img.height() as i32 / 20);
        for k in probs.topk(topk).iter() {
            let legend = format!("{}: {:.4}", k.2.as_ref().unwrap_or(&k.0.to_string()), k.1);
            let scale_dy = img.width().max(img.height()) as f32 / 30.0;
            let scale = PxScale::from(scale_dy);
            let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend);
            let text_h = text_h + text_h / 3;
            y += text_h as i32;
            imageproc::drawing::draw_filled_rect_mut(
                img,
                imageproc::rect::Rect::at(x, y).of_size(text_w, text_h),
                image::Rgb(self.get_color(k.0).into()),
            );
            imageproc::drawing::draw_text_mut(
                img,
                image::Rgb((0, 0, 0).into()),
                x,
                y - (scale_dy / self.scale_).floor() as i32 + 2,
                scale,
                &self.font,
                &legend,
            );
        }
    }

    pub fn plot_keypoints(&self, img: &mut RgbImage, keypoints: &[Vec<Keypoint>]) {
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

    fn load_font(path: Option<&str>) -> Result<FontVec> {
        let path_font = match path {
            None => auto_load("Arial.ttf")?,
            Some(p) => p.into(),
        };
        let buffer = std::fs::read(path_font)?;
        Ok(FontVec::try_from_vec(buffer.to_owned()).unwrap())
    }

    pub fn get_color(&self, n: usize) -> (u8, u8, u8) {
        Self::color_palette()[n % Self::color_palette().len()]
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
