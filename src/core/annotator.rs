use crate::{
    auto_load, colormap256, string_now, Bbox, Keypoint, Mask, Mbr, Polygon, Prob, CHECK_MARK,
    CROSS_MARK, Y,
};
use ab_glyph::{FontVec, PxScale};
use anyhow::Result;
use image::{DynamicImage, GenericImage, Rgba, RgbaImage};
use imageproc::map::map_colors;

/// Annotator for struct `Y`
#[derive(Debug)]
pub struct Annotator {
    font: FontVec,
    _scale: f32, // Cope with ab_glyph & imageproc=0.24.0
    scale_dy: f32,
    saveout: Option<String>,
    decimal_places: usize,

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
    bboxes_thickness: u8,

    // About keypoints
    without_keypoints: bool,
    with_keypoints_conf: bool,
    with_keypoints_name: bool,
    without_keypoints_text_bg: bool,
    keypoints_text_color: Rgba<u8>,
    skeletons: Option<Vec<(usize, usize)>>,
    keypoints_radius: usize,
    keypoints_palette: Option<Vec<(u8, u8, u8, u8)>>,

    // About polygons
    without_polygons: bool,
    without_contours: bool,
    with_polygons_conf: bool,
    with_polygons_name: bool,
    with_polygons_text_bg: bool,
    polygons_text_color: Rgba<u8>,
    polygons_alpha: u8,
    contours_color: Rgba<u8>,

    // About masks
    without_masks: bool,
    colormap: Option<[[u8; 3]; 256]>,

    // About probs
    probs_topk: usize,
}

impl Default for Annotator {
    fn default() -> Self {
        Self {
            font: Self::load_font(None).unwrap(),
            _scale: 6.666667,
            scale_dy: 28.,
            polygons_alpha: 179,
            saveout: None,
            decimal_places: 4,
            without_bboxes: false,
            without_bboxes_conf: false,
            without_bboxes_name: false,
            bboxes_text_color: Rgba([0, 0, 0, 255]),
            bboxes_thickness: 1,
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
            without_polygons: false,
            without_contours: false,
            contours_color: Rgba([255, 255, 255, 255]),
            with_polygons_name: false,
            with_polygons_conf: false,
            with_polygons_text_bg: false,
            polygons_text_color: Rgba([255, 255, 255, 255]),
            probs_topk: 5usize,
            without_masks: false,
            colormap: None,
        }
    }
}

impl Annotator {
    pub fn with_decimal_places(mut self, x: usize) -> Self {
        self.decimal_places = x;
        self
    }

    /// Plotting BBOXes or not
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

    pub fn withbboxes_thickness(mut self, thickness: u8) -> Self {
        self.bboxes_thickness = thickness;
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

    /// Plotting MBRs or not
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

    pub fn without_polygons(mut self, x: bool) -> Self {
        self.without_polygons = x;
        self
    }

    pub fn without_contours(mut self, x: bool) -> Self {
        self.without_contours = x;
        self
    }

    pub fn with_polygons_conf(mut self, x: bool) -> Self {
        self.with_polygons_conf = x;
        self
    }

    pub fn with_polygons_name(mut self, x: bool) -> Self {
        self.with_polygons_name = x;
        self
    }

    pub fn with_polygons_text_bg(mut self, x: bool) -> Self {
        self.with_polygons_text_bg = x;
        self
    }

    pub fn with_colormap(mut self, x: &str) -> Self {
        let x = match x {
            "turbo" | "Turbo" | "TURBO" => colormap256::TURBO,
            "inferno" | "Inferno" | "INFERNO" => colormap256::INFERNO,
            "plasma" | "Plasma" | "PLASMA" => colormap256::PLASMA,
            "viridis" | "Viridis" | "VIRIDIS" => colormap256::VIRIDIS,
            "magma" | "Magma" | "MAGMA" => colormap256::MAGMA,
            "bentcoolwarm" | "BentCoolWarm" | "BENTCOOLWARM" => colormap256::BENTCOOLWARM,
            "blackbody" | "BlackBody" | "BLACKBODY" => colormap256::BLACKBODY,
            "extendedkindLmann" | "ExtendedKindLmann" | "EXTENDEDKINDLMANN" => {
                colormap256::EXTENDEDKINDLMANN
            }
            "kindlmann" | "KindLmann" | "KINDLMANN" => colormap256::KINDLMANN,
            "smoothcoolwarm" | "SmoothCoolWarm" | "SMOOTHCOOLWARM" => colormap256::SMOOTHCOOLWARM,
            _ => todo!(),
        };
        self.colormap = Some(x);
        self
    }

    pub fn with_polygons_text_color(mut self, rgba: [u8; 4]) -> Self {
        self.polygons_text_color = Rgba(rgba);
        self
    }

    pub fn with_polygons_alpha(mut self, x: u8) -> Self {
        self.polygons_alpha = x;
        self
    }

    pub fn with_polygons_text_bg_alpha(mut self, x: u8) -> Self {
        self.polygons_text_color.0[3] = x;
        self
    }

    pub fn with_contours_color(mut self, rgba: [u8; 4]) -> Self {
        self.contours_color = Rgba(rgba);
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

    /// Save annotated images to `runs` folder
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

    /// Annotate images
    pub fn annotate(&self, imgs: &[DynamicImage], ys: &[Y]) {
        for (img, y) in imgs.iter().zip(ys.iter()) {
            let mut img_rgba = img.to_rgba8();

            // polygons
            if !self.without_polygons {
                if let Some(xs) = &y.polygons() {
                    self.plot_polygons(&mut img_rgba, xs)
                }
            }

            // bboxes
            if !self.without_bboxes {
                if let Some(xs) = &y.bboxes() {
                    self.plot_bboxes(&mut img_rgba, xs)
                }
            }

            // mbrs
            if !self.without_mbrs {
                if let Some(xs) = &y.mbrs() {
                    self.plot_mbrs(&mut img_rgba, xs)
                }
            }

            // keypoints
            if !self.without_keypoints {
                if let Some(xs) = &y.keypoints() {
                    self.plot_keypoints(&mut img_rgba, xs)
                }
            }

            // probs
            if let Some(xs) = &y.probs() {
                self.plot_probs(&mut img_rgba, xs)
            }

            // masks
            if !self.without_masks {
                if let Some(xs) = &y.masks() {
                    self.plot_masks(&mut img_rgba, xs)
                }
            }

            // save
            if let Some(saveout) = &self.saveout {
                self.save(&img_rgba, saveout);
            }
        }
    }

    /// Plot bounding bboxes and labels
    pub fn plot_bboxes(&self, img: &mut RgbaImage, bboxes: &[Bbox]) {
        for bbox in bboxes.iter() {
            // bbox
            if self.bboxes_thickness <= 1 {
                imageproc::drawing::draw_hollow_rect_mut(
                    img,
                    imageproc::rect::Rect::at(
                        bbox.xmin().round() as i32,
                        bbox.ymin().round() as i32,
                    )
                    .of_size(bbox.width().round() as u32, bbox.height().round() as u32),
                    image::Rgba(self.get_color(bbox.id() as usize).into()),
                );
            } else {
                for i in 0..self.bboxes_thickness {
                    imageproc::drawing::draw_hollow_rect_mut(
                        img,
                        imageproc::rect::Rect::at(
                            (bbox.xmin().round() as i32) - (i as i32),
                            (bbox.ymin().round() as i32) - (i as i32),
                        )
                        .of_size(
                            (bbox.width().round() as u32) + (2 * i as u32),
                            (bbox.height().round() as u32) + (2 * i as u32),
                        ),
                        image::Rgba(self.get_color(bbox.id() as usize).into()),
                    );
                }
            }

            // label
            if !self.without_bboxes_name || !self.without_bboxes_conf {
                let label = bbox.label(
                    !self.without_bboxes_name,
                    !self.without_bboxes_conf,
                    self.decimal_places,
                );
                self.put_text(
                    img,
                    &label,
                    bbox.xmin(),
                    bbox.ymin(),
                    image::Rgba(self.get_color(bbox.id() as usize).into()),
                    self.bboxes_text_color,
                    self.without_bboxes_text_bg,
                );
            }
        }
    }

    /// Plot minimum bounding rectangle and labels
    pub fn plot_mbrs(&self, img: &mut RgbaImage, mbrs: &[Mbr]) {
        for mbr in mbrs.iter() {
            // mbr
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

            // label
            if !self.without_mbrs_name || !self.without_mbrs_conf {
                let label = mbr.label(
                    !self.without_mbrs_name,
                    !self.without_mbrs_conf,
                    self.decimal_places,
                );
                self.put_text(
                    img,
                    &label,
                    mbr.top().x as f32,
                    mbr.top().y as f32,
                    image::Rgba(self.get_color(mbr.id() as usize).into()),
                    self.mbrs_text_color,
                    self.without_mbrs_text_bg,
                );
            }
        }
    }

    /// Plot polygons(hollow & filled) and labels
    pub fn plot_polygons(&self, img: &mut RgbaImage, polygons: &[Polygon]) {
        let mut convas = img.clone();
        for polygon in polygons.iter() {
            // filled
            let polygon_i32 = polygon
                .polygon()
                .exterior()
                .points()
                .take(if polygon.is_closed() {
                    polygon.count() - 1
                } else {
                    polygon.count()
                })
                .map(|p| imageproc::point::Point::new(p.x() as i32, p.y() as i32))
                .collect::<Vec<_>>();
            let mut color_ = self.get_color(polygon.id() as usize);
            color_.3 = self.polygons_alpha;
            imageproc::drawing::draw_polygon_mut(&mut convas, &polygon_i32, Rgba(color_.into()));

            // contour
            if !self.without_contours {
                let polygon_f32 = polygon
                    .polygon()
                    .exterior()
                    .points()
                    .take(if polygon.is_closed() {
                        polygon.count() - 1
                    } else {
                        polygon.count()
                    })
                    .map(|p| imageproc::point::Point::new(p.x() as f32, p.y() as f32))
                    .collect::<Vec<_>>();
                imageproc::drawing::draw_hollow_polygon_mut(img, &polygon_f32, self.contours_color);
            }
        }
        image::imageops::overlay(img, &convas, 0, 0);

        // labels on top
        if self.with_polygons_name || self.with_polygons_conf {
            for polygon in polygons.iter() {
                if let Some((x, y)) = polygon.centroid() {
                    let label = polygon.label(
                        self.with_polygons_name,
                        self.with_polygons_conf,
                        self.decimal_places,
                    );
                    self.put_text(
                        img,
                        &label,
                        x,
                        y,
                        image::Rgba(self.get_color(polygon.id() as usize).into()),
                        self.polygons_text_color,
                        !self.with_polygons_text_bg,
                    );
                }
            }
        }
    }

    /// Plot keypoints and texts
    pub fn plot_keypoints(&self, img: &mut RgbaImage, keypoints: &[Vec<Keypoint>]) {
        for kpts in keypoints.iter() {
            for (i, kpt) in kpts.iter().enumerate() {
                if kpt.confidence() == 0.0 {
                    continue;
                }

                // keypoint
                let color = match &self.keypoints_palette {
                    None => self.get_color(i),
                    Some(keypoints_palette) => keypoints_palette[i],
                };
                imageproc::drawing::draw_filled_circle_mut(
                    img,
                    (kpt.x() as i32, kpt.y() as i32),
                    self.keypoints_radius as i32,
                    image::Rgba(color.into()),
                );

                // label
                if self.with_keypoints_name || self.with_keypoints_conf {
                    let label = kpt.label(
                        self.with_keypoints_name,
                        self.with_keypoints_conf,
                        self.decimal_places,
                    );
                    self.put_text(
                        img,
                        &label,
                        kpt.x(),
                        kpt.y(),
                        image::Rgba(self.get_color(kpt.id() as usize).into()),
                        self.keypoints_text_color,
                        self.without_keypoints_text_bg,
                    );
                }
            }

            // skeletons
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

    /// Plot masks
    pub fn plot_masks(&self, img: &mut RgbaImage, masks: &[Mask]) {
        let (w, h) = img.dimensions();
        // let hstack = w < h;
        let hstack = true;
        let scale = 2;
        let size = (masks.len() + 1) as u32;

        // convas
        let convas = img.clone();
        let mut convas = image::DynamicImage::from(convas);
        if hstack {
            convas = convas.resize_exact(
                w,
                h / scale * (size / scale),
                image::imageops::FilterType::CatmullRom,
            );
        } else {
            convas = convas.resize_exact(
                w / scale,
                h * size / scale,
                image::imageops::FilterType::CatmullRom,
            );
        }
        for x in 0..convas.width() {
            for y in 0..convas.height() {
                convas.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }

        // place original
        let im_ori = img.clone();
        let im_ori = image::DynamicImage::from(im_ori);
        let im_ori = im_ori.resize_exact(
            w / scale,
            h / scale,
            image::imageops::FilterType::CatmullRom,
        );
        image::imageops::overlay(&mut convas, &im_ori, 0, 0);

        // place masks
        for (i, mask) in masks.iter().enumerate() {
            let i = i + 1;
            let luma = if let Some(colormap) = self.colormap {
                let luma = map_colors(mask.mask(), |p| {
                    let x = p[0];
                    image::Rgb(colormap[x as usize])
                });
                image::DynamicImage::from(luma)
            } else {
                mask.mask().to_owned()
            };
            let luma = luma.resize_exact(
                w / scale,
                h / scale,
                image::imageops::FilterType::CatmullRom,
            );
            if hstack {
                let pos_x = (i as u32 % scale) * luma.width();
                let pos_y = (i as u32 / scale) * luma.height();
                image::imageops::overlay(&mut convas, &luma, pos_x as i64, pos_y as i64);
            } else {
                let pos_x = 0;
                let pos_y = i as u32 * luma.height();
                image::imageops::overlay(&mut convas, &luma, pos_x as i64, pos_y as i64);
            }
        }
        *img = convas.into_rgba8();
    }

    /// Plot probs
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

    /// Helper for putting texts
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

    /// Load custom font
    fn load_font(path: Option<&str>) -> Result<FontVec> {
        let path_font = match path {
            None => auto_load("Arial.ttf", Some("fonts"))?,
            Some(p) => p.into(),
        };
        let buffer = std::fs::read(path_font)?;
        Ok(FontVec::try_from_vec(buffer.to_owned()).unwrap())
    }

    /// Pick color from pallette
    pub fn get_color(&self, n: usize) -> (u8, u8, u8, u8) {
        Self::color_palette()[n % Self::color_palette().len()]
    }

    /// Color pallette
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
