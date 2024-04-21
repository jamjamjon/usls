use crate::{auto_load, string_now, Bbox, Keypoint, Mask, Mbr, Prob, CHECK_MARK, CROSS_MARK, Y};
use ab_glyph::{FontVec, PxScale};
use anyhow::Result;
use image::{DynamicImage, Rgba, RgbaImage};

/// Annotator for struct `Y`
#[derive(Debug)]
pub struct Annotator {
    font: FontVec,
    _scale: f32, // Cope with ab_glyph & imageproc=0.24.0
    scale_dy: f32,
    saveout: Option<String>,
    // About mbrs
    without_mbrs: bool,
    without_mbrs_conf: bool,
    without_mbrs_name: bool,
    without_mbrs_text_bg: bool,
    mbrs_text_color: Rgba<u8>,
    // About bboxes
    without_bboxes: bool,
    without_bboxes_conf: bool,
    without_bboxes_name: bool,
    without_bboxes_text_bg: bool,
    bboxes_text_color: Rgba<u8>,
    // About keypoints
    without_keypoints: bool,
    with_keypoints_conf: bool,
    with_keypoints_name: bool,
    without_keypoints_text_bg: bool,
    keypoints_text_color: Rgba<u8>,
    skeletons: Option<Vec<(usize, usize)>>,
    keypoints_radius: usize,
    keypoints_palette: Option<Vec<(u8, u8, u8, u8)>>,
    // About masks
    without_masks: bool,
    without_polygons: bool,
    with_masks_conf: bool,
    with_masks_name: bool,
    with_masks_text_bg: bool,
    masks_text_color: Rgba<u8>,
    masks_alpha: u8,
    polygon_color: Rgba<u8>,
    // About probs
    probs_topk: usize,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            font: Self::load_font(None).unwrap(),
            _scale: 6.666667,
            scale_dy: 28.,
            masks_alpha: 179,
            saveout: None,
            without_bboxes: false,
            without_bboxes_conf: false,
            without_bboxes_name: false,
            bboxes_text_color: Rgba([0, 0, 0, 255]),
            without_bboxes_text_bg: false,
            without_mbrs: false,
            without_mbrs_conf: false,
            without_mbrs_name: false,
            without_mbrs_text_bg: false,
            mbrs_text_color: Rgba([0, 0, 0, 255]),
            without_keypoints: false,
            with_keypoints_conf: false,
            with_keypoints_name: false,
            keypoints_radius: 3,
            skeletons: None,
            keypoints_palette: None,
            without_keypoints_text_bg: false,
            keypoints_text_color: Rgba([0, 0, 0, 255]),
            without_masks: false,
            without_polygons: false,
            polygon_color: Rgba([255, 255, 255, 255]),
            with_masks_name: false,
            with_masks_conf: false,
            with_masks_text_bg: false,
            masks_text_color: Rgba([255, 255, 255, 255]),
            probs_topk: 5usize,
        }
    }
}

impl Annotator {
    pub fn without_bboxes(mut self, x: bool) -> Self {
        self.without_bboxes = x;
        self
    }

    pub fn without_bboxes_conf(mut self, x: bool) -> Self {
        self.without_bboxes_conf = x;
        self
    }

    pub fn without_bboxes_name(mut self, x: bool) -> Self {
        self.without_bboxes_name = x;
        self
    }

    pub fn without_bboxes_text_bg(mut self, x: bool) -> Self {
        self.without_bboxes_text_bg = x;
        self
    }

    pub fn with_bboxes_text_bg_alpha(mut self, x: u8) -> Self {
        self.bboxes_text_color.0[3] = x;
        self
    }

    pub fn with_bboxes_text_color(mut self, rgba: [u8; 4]) -> Self {
        self.bboxes_text_color = Rgba(rgba);
        self
    }

    pub fn without_keypoints(mut self, x: bool) -> Self {
        self.without_keypoints = x;
        self
    }

    pub fn with_skeletons(mut self, x: &[(usize, usize)]) -> Self {
        self.skeletons = Some(x.to_vec());
        self
    }

    pub fn with_keypoints_palette(mut self, x: &[(u8, u8, u8, u8)]) -> Self {
        self.keypoints_palette = Some(x.to_vec());
        self
    }

    pub fn with_keypoints_radius(mut self, x: usize) -> Self {
        self.keypoints_radius = x;
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

    pub fn with_keypoints_text_color(mut self, rgba: [u8; 4]) -> Self {
        self.keypoints_text_color = Rgba(rgba);
        self
    }

    pub fn without_keypoints_text_bg(mut self, x: bool) -> Self {
        self.without_keypoints_text_bg = x;
        self
    }

    pub fn with_keypoints_text_bg_alpha(mut self, x: u8) -> Self {
        self.keypoints_text_color.0[3] = x;
        self
    }

    pub fn without_mbrs(mut self, x: bool) -> Self {
        self.without_mbrs = x;
        self
    }

    pub fn without_mbrs_conf(mut self, x: bool) -> Self {
        self.without_mbrs_conf = x;
        self
    }

    pub fn without_mbrs_name(mut self, x: bool) -> Self {
        self.without_mbrs_name = x;
        self
    }

    pub fn without_mbrs_text_bg(mut self, x: bool) -> Self {
        self.without_mbrs_text_bg = x;
        self
    }

    pub fn with_mbrs_text_color(mut self, rgba: [u8; 4]) -> Self {
        self.mbrs_text_color = Rgba(rgba);
        self
    }

    pub fn with_mbrs_text_bg_alpha(mut self, x: u8) -> Self {
        self.mbrs_text_color.0[3] = x;
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

    pub fn with_masks_conf(mut self, x: bool) -> Self {
        self.with_masks_conf = x;
        self
    }

    pub fn with_masks_name(mut self, x: bool) -> Self {
        self.with_masks_name = x;
        self
    }

    pub fn with_masks_text_bg(mut self, x: bool) -> Self {
        self.with_masks_text_bg = x;
        self
    }

    pub fn with_masks_text_color(mut self, rgba: [u8; 4]) -> Self {
        self.masks_text_color = Rgba(rgba);
        self
    }

    pub fn with_masks_alpha(mut self, x: u8) -> Self {
        self.masks_alpha = x;
        self
    }

    pub fn with_masks_text_bg_alpha(mut self, x: u8) -> Self {
        self.masks_text_color.0[3] = x;
        self
    }

    pub fn with_polygon_color(mut self, rgba: [u8; 4]) -> Self {
        self.polygon_color = Rgba(rgba);
        self
    }

    pub fn with_probs_topk(mut self, x: usize) -> Self {
        self.probs_topk = x;
        self
    }

    pub fn with_saveout(mut self, saveout: &str) -> Self {
        self.saveout = Some(saveout.to_string());
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

    pub fn annotate(&self, imgs: &[DynamicImage], ys: &[Y]) {
        for (img, y) in imgs.iter().zip(ys.iter()) {
            let mut img_rgb = img.to_rgba8();

            // masks
            if !self.without_masks {
                if let Some(xs) = &y.masks() {
                    self.plot_masks_and_polygons(&mut img_rgb, xs)
                }
            }

            // bboxes
            if !self.without_bboxes {
                if let Some(xs) = &y.bboxes() {
                    self.plot_bboxes(&mut img_rgb, xs)
                }
            }

            // mbrs
            if !self.without_mbrs {
                if let Some(xs) = &y.mbrs() {
                    self.plot_mbrs(&mut img_rgb, xs)
                }
            }

            // keypoints
            if !self.without_keypoints {
                if let Some(xs) = &y.keypoints() {
                    self.plot_keypoints(&mut img_rgb, xs)
                }
            }

            // probs
            if let Some(xs) = &y.probs() {
                self.plot_probs(&mut img_rgb, xs)
            }

            // save
            if let Some(saveout) = &self.saveout {
                self.save(&img_rgb, saveout);
            }
        }
    }

    pub fn plot_bboxes(&self, img: &mut RgbaImage, bboxes: &[Bbox]) {
        for bbox in bboxes.iter() {
            // bboxes
            imageproc::drawing::draw_hollow_rect_mut(
                img,
                imageproc::rect::Rect::at(bbox.xmin().round() as i32, bbox.ymin().round() as i32)
                    .of_size(bbox.width().round() as u32, bbox.height().round() as u32),
                image::Rgba(self.get_color(bbox.id() as usize).into()),
            );

            // texts
            let mut legend = String::new();
            if !self.without_bboxes_name {
                legend.push_str(&bbox.name().unwrap_or(&bbox.id().to_string()).to_string());
            }
            if !self.without_bboxes_conf {
                if !self.without_bboxes_name {
                    legend.push_str(&format!(": {:.4}", bbox.confidence()));
                } else {
                    legend.push_str(&format!("{:.4}", bbox.confidence()));
                }
            }
            self.put_text(
                img,
                legend.as_str(),
                bbox.xmin(),
                bbox.ymin(),
                image::Rgba(self.get_color(bbox.id() as usize).into()),
                self.bboxes_text_color,
                self.without_bboxes_text_bg,
            );
        }
    }

    pub fn plot_mbrs(&self, img: &mut RgbaImage, mbrs: &[Mbr]) {
        for mbr in mbrs.iter() {
            // mbrs
            for i in 0..mbr.vertices().len() {
                let p1 = mbr.vertices()[i];
                let p2 = mbr.vertices()[(i + 1) % mbr.vertices().len()];
                imageproc::drawing::draw_line_segment_mut(
                    img,
                    (p1.x.round() as f32, p1.y.round() as f32),
                    (p2.x.round() as f32, p2.y.round() as f32),
                    image::Rgba(self.get_color(mbr.id() as usize).into()),
                );
            }

            // text
            let mut legend = String::new();
            if !self.without_mbrs_name {
                legend.push_str(&mbr.name().unwrap_or(&mbr.id().to_string()).to_string());
            }
            if !self.without_mbrs_conf {
                if !self.without_mbrs_name {
                    legend.push_str(&format!(": {:.4}", mbr.confidence()));
                } else {
                    legend.push_str(&format!("{:.4}", mbr.confidence()));
                }
            }
            self.put_text(
                img,
                legend.as_str(),
                mbr.top().x as f32,
                mbr.top().y as f32,
                image::Rgba(self.get_color(mbr.id() as usize).into()),
                self.mbrs_text_color,
                self.without_mbrs_text_bg,
            );
        }
    }

    pub fn plot_masks_and_polygons(&self, img: &mut RgbaImage, masks: &[Mask]) {
        let mut convas = img.clone();
        for mask in masks.iter() {
            // masks
            let polygon_i32 = mask
                .polygon()
                .exterior()
                .points()
                .take(if mask.is_closed() {
                    mask.count() - 1
                } else {
                    mask.count()
                })
                .map(|p| imageproc::point::Point::new(p.x() as i32, p.y() as i32))
                .collect::<Vec<_>>();
            let mut mask_color = self.get_color(mask.id() as usize);
            mask_color.3 = self.masks_alpha;
            imageproc::drawing::draw_polygon_mut(
                &mut convas,
                &polygon_i32,
                Rgba(mask_color.into()),
            );

            // contours(polygons)
            if !self.without_polygons {
                let polygon_f32 = mask
                    .polygon()
                    .exterior()
                    .points()
                    .take(if mask.is_closed() {
                        mask.count() - 1
                    } else {
                        mask.count()
                    })
                    .map(|p| imageproc::point::Point::new(p.x() as f32, p.y() as f32))
                    .collect::<Vec<_>>();
                imageproc::drawing::draw_hollow_polygon_mut(img, &polygon_f32, self.polygon_color);
            }
        }
        image::imageops::overlay(img, &convas, 0, 0);

        // text on top
        for mask in masks.iter() {
            if let Some((x, y)) = mask.centroid() {
                let mut legend = String::new();
                if self.with_masks_name {
                    legend.push_str(&mask.name().unwrap_or(&mask.id().to_string()).to_string());
                }
                if self.with_masks_conf {
                    if self.with_masks_name {
                        legend.push_str(&format!(": {:.4}", mask.confidence()));
                    } else {
                        legend.push_str(&format!("{:.4}", mask.confidence()));
                    }
                }
                self.put_text(
                    img,
                    legend.as_str(),
                    x,
                    y,
                    image::Rgba(self.get_color(mask.id() as usize).into()),
                    self.masks_text_color,
                    !self.with_masks_text_bg,
                );
            }
        }
    }

    pub fn plot_probs(&self, img: &mut RgbaImage, probs: &Prob) {
        let (x, mut y) = (img.width() as i32 / 20, img.height() as i32 / 20);
        for k in probs.topk(self.probs_topk).iter() {
            let legend = format!("{}: {:.4}", k.2.as_ref().unwrap_or(&k.0.to_string()), k.1);
            let scale = PxScale::from(self.scale_dy);
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
                y - (self.scale_dy / self._scale).floor() as i32 + 2,
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

                // keypoints
                let color = match &self.keypoints_palette {
                    None => self.get_color(i + 10),
                    Some(keypoints_palette) => keypoints_palette[i],
                };
                imageproc::drawing::draw_filled_circle_mut(
                    img,
                    (kpt.x() as i32, kpt.y() as i32),
                    self.keypoints_radius as i32,
                    image::Rgba(color.into()),
                );

                // text
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
                self.put_text(
                    img,
                    legend.as_str(),
                    kpt.x(),
                    kpt.y(),
                    image::Rgba(self.get_color(kpt.id() as usize).into()),
                    self.keypoints_text_color,
                    self.without_keypoints_text_bg,
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
                        image::Rgba([255, 51, 255, 255]),
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn put_text(
        &self,
        img: &mut RgbaImage,
        legend: &str,
        x: f32,
        y: f32,
        color: Rgba<u8>,
        text_color: Rgba<u8>,
        without_text_bg: bool,
    ) {
        if !legend.is_empty() {
            let scale = PxScale::from(self.scale_dy);
            let (text_w, text_h) = imageproc::drawing::text_size(scale, &self.font, legend);
            let text_h = text_h + text_h / 3;
            let top = if y > text_h as f32 {
                (y.round() as u32 - text_h) as i32
            } else {
                (text_h - y.round() as u32) as i32
            };
            let mut left = x as i32;
            if left + text_w as i32 > img.width() as i32 {
                left = img.width() as i32 - text_w as i32;
            }

            // text bbox
            if !without_text_bg {
                imageproc::drawing::draw_filled_rect_mut(
                    img,
                    imageproc::rect::Rect::at(left, top).of_size(text_w, text_h),
                    color,
                );
            }

            // text
            imageproc::drawing::draw_text_mut(
                img,
                text_color,
                left,
                top - (self.scale_dy / self._scale).floor() as i32 + 2,
                scale,
                &self.font,
                legend,
            );
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

    fn color_palette() -> [(u8, u8, u8, u8); 20] {
        [
            (0, 255, 127, 255),   // spring green
            (255, 105, 180, 255), // hot pink
            (255, 99, 71, 255),   // tomato
            (255, 215, 0, 255),   // glod
            (188, 143, 143, 255), // rosy brown
            (0, 191, 255, 255),   // deep sky blue
            (143, 188, 143, 255), // dark sea green
            (238, 130, 238, 255), // violet
            (154, 205, 50, 255),  // yellow green
            (205, 133, 63, 255),  // peru
            (30, 144, 255, 255),  // dodger blue
            (112, 128, 144, 255), // slate gray
            (127, 255, 212, 255), // aqua marine
            (51, 153, 255, 255),  // blue
            (0, 255, 255, 255),   // cyan
            (138, 43, 226, 255),  // blue violet
            (165, 42, 42, 255),   // brown
            (216, 191, 216, 255), // thistle
            (240, 255, 255, 255), // azure
            (95, 158, 160, 255),  // cadet blue
        ]
    }
}
