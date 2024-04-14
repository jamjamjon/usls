use crate::{auto_load, string_now, Bbox, Embedding, Keypoint, Mask, Ys, CHECK_MARK, CROSS_MARK};
use ab_glyph::{FontVec, PxScale};
use anyhow::Result;
use image::{DynamicImage, Rgba, RgbaImage};

#[derive(Debug)]
pub struct Annotator {
    font: ab_glyph::FontVec,
    scale_: f32, // Cope with ab_glyph & imageproc=0.24.0
    skeletons: Option<Vec<(usize, usize)>>,
    saveout: Option<String>,
    mask_alpha: u8,
    polygon_color: Rgba<u8>,
    without_conf: bool,
    without_name: bool,
    with_keypoints_conf: bool,
    with_keypoints_name: bool,
    with_masks_name: bool,
    without_bboxes: bool,
    without_masks: bool,
    without_polygons: bool,
    without_keypoints: bool,
    keypoint_radius: usize,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            font: Self::load_font(None).unwrap(),
            scale_: 6.666667,
            mask_alpha: 179,
            polygon_color: Rgba([255, 255, 255, 255]),
            skeletons: None,
            saveout: None,
            without_conf: false,
            without_name: false,
            with_keypoints_conf: false,
            with_keypoints_name: false,
            with_masks_name: false,
            without_bboxes: false,
            without_masks: false,
            without_polygons: false,
            without_keypoints: false,
            keypoint_radius: 3,
        }
    }
}

impl Annotator {
    pub fn with_keypoint_radius(mut self, x: usize) -> Self {
        self.keypoint_radius = x;
        self
    }

    pub fn without_conf(mut self, x: bool) -> Self {
        self.without_conf = x;
        self
    }

    pub fn without_name(mut self, x: bool) -> Self {
        self.without_name = x;
        self
    }

    pub fn with_keypoints_conf(mut self, x: bool) -> Self {
        self.with_keypoints_conf = x;
        self
    }

    pub fn with_keypoints_name(mut self, x: bool) -> Self {
        self.with_keypoints_name = x;
        self
    }

    pub fn with_masks_name(mut self, x: bool) -> Self {
        self.with_masks_name = x;
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

    pub fn with_mask_alpha(mut self, x: u8) -> Self {
        self.mask_alpha = x;
        self
    }

    pub fn with_polygon_color(mut self, rgba: [u8; 4]) -> Self {
        self.polygon_color = Rgba(rgba);
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

    pub fn with_skeletons(mut self, skeletons: &[(usize, usize)]) -> Self {
        self.skeletons = Some(skeletons.to_vec());
        self
    }

    pub fn with_font(mut self, path: &str) -> Self {
        self.font = Self::load_font(Some(path)).unwrap();
        self
    }

    pub fn save(&self, image: &RgbaImage, saveout: &str) {
        let mut saveout = std::path::PathBuf::from("runs").join(saveout);
        if !saveout.exists() {
            std::fs::create_dir_all(&saveout).unwrap();
        }
        saveout.push(string_now("-"));
        let saveout = format!("{}.png", saveout.to_str().unwrap());
        match image.save(&saveout) {
            Err(err) => println!("{} Saving failed: {:?}", CROSS_MARK, err),
            Ok(_) => println!("{} Annotated image saved to: {}", CHECK_MARK, saveout),
        }
    }

    pub fn annotate(&self, imgs: &[DynamicImage], ys: &[Ys]) {
        for (img, y) in imgs.iter().zip(ys.iter()) {
            let mut img_rgb = img.to_rgba8();

            // masks
            if !self.without_polygons {
                if let Some(xs) = &y.masks {
                    self.plot_polygons(&mut img_rgb, xs)
                }
            }

            // bboxes
            if !self.without_bboxes {
                if let Some(xs) = &y.bboxes {
                    self.plot_bboxes(&mut img_rgb, xs)
                }
            }

            // keypoints
            if !self.without_keypoints {
                if let Some(xs) = &y.keypoints {
                    self.plot_keypoints(&mut img_rgb, xs)
                }
            }

            // probs
            if let Some(xs) = &y.probs {
                self.plot_probs(&mut img_rgb, xs)
            }

            if let Some(saveout) = &self.saveout {
                self.save(&img_rgb, saveout);
            }
        }
    }

    pub fn plot_bboxes(&self, img: &mut RgbaImage, bboxes: &[Bbox]) {
        for bbox in bboxes.iter() {
            imageproc::drawing::draw_hollow_rect_mut(
                img,
                imageproc::rect::Rect::at(bbox.xmin().round() as i32, bbox.ymin().round() as i32)
                    .of_size(bbox.width().round() as u32, bbox.height().round() as u32),
                image::Rgba(self.get_color(bbox.id()).into()),
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
            if !legend.is_empty() {
                let scale_dy = img.width().max(img.height()) as f32 / 40.0;
                let scale = PxScale::from(scale_dy);
                let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend); // u32
                let text_h = text_h + text_h / 3;
                let top = if bbox.ymin() > text_h as f32 {
                    (bbox.ymin().round() as u32 - text_h) as i32
                } else {
                    (text_h - bbox.ymin().round() as u32) as i32
                };
                let mut left = bbox.xmin() as i32;
                if left + text_w as i32 > img.width() as i32 {
                    left = img.width() as i32 - text_w as i32;
                }
                imageproc::drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
                    image::Rgba(self.get_color(bbox.id()).into()),
                );
                imageproc::drawing::draw_text_mut(
                    img,
                    image::Rgba([0, 0, 0, 255]),
                    left,
                    top - (scale_dy / self.scale_).floor() as i32 + 2,
                    scale,
                    &self.font,
                    &legend,
                );
            }
        }
    }

    pub fn plot_polygons(&self, img: &mut RgbaImage, masks: &[Mask]) {
        let mut convas = img.clone();
        for mask in masks.iter() {
            // mask
            let mut polygon_i32 = mask
                .polygon
                .points
                .iter()
                .map(|p| imageproc::point::Point::new(p.x as i32, p.y as i32))
                .collect::<Vec<_>>();
            if polygon_i32.first() == polygon_i32.last() {
                polygon_i32.pop();
            }
            let mut mask_color = self.get_color(mask.id);
            mask_color.3 = self.mask_alpha;
            imageproc::drawing::draw_polygon_mut(
                &mut convas,
                &polygon_i32,
                Rgba(mask_color.into()),
            );

            // contour
            let polygon_f32 = mask
                .polygon
                .points
                .iter()
                .map(|p| imageproc::point::Point::new(p.x, p.y))
                .collect::<Vec<_>>();
            imageproc::drawing::draw_hollow_polygon_mut(img, &polygon_f32, self.polygon_color);

            // text
            let mut legend = String::new();
            if self.with_masks_name {
                legend.push_str(&mask.name().unwrap_or(&mask.id().to_string()).to_string());
            }
            if !legend.is_empty() {
                let scale_dy = img.width().max(img.height()) as f32 / 60.0;
                let scale = PxScale::from(scale_dy);
                let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, &legend); // u32
                let text_h = text_h + text_h / 3;
                let bbox = mask.polygon.find_min_rect();
                let top = (bbox.cy().round() as u32 - text_h) as i32;
                let mut left = (bbox.cx() as i32 - text_w as i32 / 2).max(0);
                if left + text_w as i32 > img.width() as i32 {
                    left = img.width() as i32 - text_w as i32;
                }
                imageproc::drawing::draw_filled_rect_mut(
                    &mut convas,
                    imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
                    image::Rgba(self.get_color(mask.id()).into()),
                );
                imageproc::drawing::draw_text_mut(
                    &mut convas,
                    image::Rgba([0, 0, 0, 255]),
                    left,
                    top - (scale_dy / self.scale_).floor() as i32 + 2,
                    scale,
                    &self.font,
                    &legend,
                );
            }
        }
        image::imageops::overlay(img, &convas, 0, 0);
    }

    pub fn plot_probs(&self, img: &mut RgbaImage, probs: &Embedding) {
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
                image::Rgba(self.get_color(k.0).into()),
            );
            imageproc::drawing::draw_text_mut(
                img,
                image::Rgba([0, 0, 0, 255]),
                x,
                y - (scale_dy / self.scale_).floor() as i32 + 2,
                scale,
                &self.font,
                &legend,
            );
        }
    }

    pub fn plot_keypoints(&self, img: &mut RgbaImage, keypoints: &[Vec<Keypoint>]) {
        for kpts in keypoints.iter() {
            for (i, kpt) in kpts.iter().enumerate() {
                if kpt.confidence() == 0.0 {
                    continue;
                }
                imageproc::drawing::draw_filled_circle_mut(
                    img,
                    (kpt.x() as i32, kpt.y() as i32),
                    self.keypoint_radius as i32,
                    image::Rgba(self.get_color(i + 10).into()),
                );
                let mut legend = String::new();
                if self.with_keypoints_name {
                    legend.push_str(&kpt.name().unwrap_or(&kpt.id().to_string()).to_string());
                }
                if self.with_keypoints_conf {
                    if self.with_keypoints_name {
                        legend.push_str(&format!(": {:.4}", kpt.confidence()));
                    } else {
                        legend.push_str(&format!("{:.4}", kpt.confidence()));
                    }
                }
                if !legend.is_empty() {
                    let scale_dy = img.width().max(img.height()) as f32 / 80.0;
                    let scale = PxScale::from(scale_dy);
                    let (text_w, text_h) =
                        imageproc::drawing::text_size(scale, &self.font, &legend); // u32
                    let text_h = text_h + text_h / 3;
                    let top = if kpt.y() > text_h as f32 {
                        (kpt.y().round() as u32 - text_h - self.keypoint_radius as u32) as i32
                    } else {
                        (text_h - self.keypoint_radius as u32 - kpt.y().round() as u32) as i32
                    };
                    let mut left =
                        (kpt.x() as i32 - self.keypoint_radius as i32 - text_w as i32 / 2).max(0);
                    if left + text_w as i32 > img.width() as i32 {
                        left = img.width() as i32 - text_w as i32;
                    }
                    imageproc::drawing::draw_filled_rect_mut(
                        img,
                        imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
                        image::Rgba(self.get_color(kpt.id() as usize).into()),
                    );
                    imageproc::drawing::draw_text_mut(
                        img,
                        image::Rgba([0, 0, 0, 255]),
                        left,
                        top - (scale_dy / self.scale_).floor() as i32 + 2,
                        scale,
                        &self.font,
                        &legend,
                    );
                }
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
                        image::Rgba([255, 51, 255, 255]),
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

    pub fn get_color(&self, n: usize) -> (u8, u8, u8, u8) {
        Self::color_palette()[n % Self::color_palette().len()]
    }

    fn color_palette() -> Vec<(u8, u8, u8, u8)> {
        vec![
            (0, 255, 0, 255),
            (255, 128, 0, 255),
            (0, 0, 255, 255),
            (255, 153, 51, 255),
            (255, 0, 0, 255),
            (255, 51, 255, 255),
            (102, 178, 255, 255),
            (51, 153, 255, 255),
            (255, 51, 51, 255),
            (153, 255, 153, 255),
            (102, 255, 102, 255),
            (153, 204, 255, 255),
            (255, 153, 153, 255),
            (255, 178, 102, 255),
            (230, 230, 0, 255),
            (255, 153, 255, 255),
            (255, 102, 255, 255),
            (255, 102, 102, 255),
            (51, 255, 51, 255),
            (255, 255, 255, 255),
        ]
    }
}
