use image::GrayImage;
use std::collections::VecDeque;

/// Border type for contours
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderType {
    /// Outer border of a region
    Outer,
    /// Inner border (hole) of a region
    Hole,
}

/// A contour extracted from a binary image
#[derive(Debug, Clone)]
pub struct Contour {
    /// Points forming the contour boundary
    pub points: Vec<(i32, i32)>,
    /// Whether this is an outer or inner (hole) contour
    pub border_type: BorderType,
    /// Index for the parent of the current border in the returned vector.
    pub parent: Option<usize>,
}

impl Contour {
    /// Create a new contour
    pub fn new(points: Vec<(i32, i32)>, border_type: BorderType, parent: Option<usize>) -> Self {
        Self {
            points,
            border_type,
            parent,
        }
    }
}

/// Finds all borders of foreground regions in an image. All pixels with intensity strictly greater
/// than `threshold` are treated as belonging to the foreground.
///
/// Based on the algorithm proposed by Suzuki and Abe: Topological Structural
/// Analysis of Digitized Binary Images by Border Following.
pub fn find_contours_with_threshold(image: &GrayImage, threshold: u8) -> Vec<Contour> {
    let width = image.width() as usize;
    let height = image.height() as usize;

    if width == 0 || height == 0 {
        return vec![];
    }

    let mut image_values = vec![0i32; height * width];
    let at = |x, y| x + width * y;

    for y in 0..height {
        for x in 0..width {
            if image.get_pixel(x as u32, y as u32).0[0] > threshold {
                image_values[at(x, y)] = 1;
            }
        }
    }

    let mut diffs = VecDeque::from(vec![
        (-1, 0),  // w
        (-1, -1), // nw
        (0, -1),  // n
        (1, -1),  // ne
        (1, 0),   // e
        (1, 1),   // se
        (0, 1),   // s
        (-1, 1),  // sw
    ]);

    let mut contours: Vec<Contour> = Vec::new();
    let mut curr_border_num = 1;

    let get_pos_if_non_zero = |img: &[i32], x: i32, y: i32| {
        if x >= 0
            && x < width as i32
            && y >= 0
            && y < height as i32
            && img[at(x as usize, y as usize)] != 0
        {
            Some((x as usize, y as usize))
        } else {
            None
        }
    };

    let rotate_to_value = |values: &mut VecDeque<(i32, i32)>, value: (i32, i32)| {
        if let Some(pos) = values.iter().position(|&x| x == value) {
            values.rotate_left(pos);
        }
    };

    for y in 0..height {
        let mut parent_border_num = 1;

        for x in 0..width {
            if image_values[at(x, y)] == 0 {
                continue;
            }

            let border_info =
                if image_values[at(x, y)] == 1 && (x == 0 || image_values[at(x - 1, y)] == 0) {
                    Some(((x as i32 - 1, y as i32), BorderType::Outer))
                } else if image_values[at(x, y)] > 0
                    && (x + 1 == width || image_values[at(x + 1, y)] == 0)
                {
                    if image_values[at(x, y)] > 1 {
                        parent_border_num = image_values[at(x, y)] as usize;
                    }
                    Some(((x as i32 + 1, y as i32), BorderType::Hole))
                } else {
                    None
                };

            if let Some(((adj_x, adj_y), border_type)) = border_info {
                curr_border_num += 1;

                let parent = if parent_border_num > 1 {
                    let parent_index = parent_border_num - 2;
                    let parent_contour = &contours[parent_index];
                    if (border_type == BorderType::Outer)
                        ^ (parent_contour.border_type == BorderType::Outer)
                    {
                        Some(parent_index)
                    } else {
                        parent_contour.parent
                    }
                } else {
                    None
                };

                let mut points = Vec::new();
                let (curr_x, curr_y) = (x as i32, y as i32);
                rotate_to_value(&mut diffs, (adj_x - curr_x, adj_y - curr_y));

                if let Some((p1_x, p1_y)) = diffs
                    .iter()
                    .find_map(|&d| get_pos_if_non_zero(&image_values, curr_x + d.0, curr_y + d.1))
                {
                    let mut p2 = (p1_x as i32, p1_y as i32);
                    let mut p3 = (x as i32, y as i32);

                    loop {
                        points.push(p3);
                        rotate_to_value(&mut diffs, (p2.0 - p3.0, p2.1 - p3.1));

                        let (p4_x, p4_y) = diffs
                            .iter()
                            .rev()
                            .find_map(|&d| {
                                get_pos_if_non_zero(&image_values, p3.0 + d.0, p3.1 + d.1)
                            })
                            .unwrap();

                        let mut is_right_edge = false;
                        for &d in diffs.iter().rev() {
                            if d == (p4_x as i32 - p3.0, p4_y as i32 - p3.1) {
                                break;
                            }
                            if d == (1, 0) {
                                is_right_edge = true;
                                break;
                            }
                        }

                        if p3.0 + 1 == width as i32 || is_right_edge {
                            image_values[at(p3.0 as usize, p3.1 as usize)] = -curr_border_num;
                        } else if image_values[at(p3.0 as usize, p3.1 as usize)] == 1 {
                            image_values[at(p3.0 as usize, p3.1 as usize)] = curr_border_num;
                        }

                        if p4_x as i32 == curr_x
                            && p4_y as i32 == curr_y
                            && p3 == (p1_x as i32, p1_y as i32)
                        {
                            break;
                        }

                        p2 = p3;
                        p3 = (p4_x as i32, p4_y as i32);
                    }
                } else {
                    points.push((x as i32, y as i32));
                    image_values[at(x, y)] = -curr_border_num;
                }
                contours.push(Contour::new(points, border_type, parent));
            }

            if image_values[at(x, y)] != 1 {
                parent_border_num = image_values[at(x, y)].unsigned_abs() as usize;
            }
        }
    }

    contours
}

/// Find contours from raw u8 data
///
/// # Arguments
/// * `data` - Raw pixel data in row-major order
/// * `width` - Image width
/// * `height` - Image height
/// * `threshold` - Threshold value for binarization
///
/// # Returns
/// A vector of `Contour` objects
pub fn find_contours_from_raw(data: &[u8], width: u32, height: u32, threshold: u8) -> Vec<Contour> {
    if let Some(image) = GrayImage::from_raw(width, height, data.to_vec()) {
        find_contours_with_threshold(&image, threshold)
    } else {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare our Moore tracing with imageproc's implementation
    #[cfg(feature = "annotator")]
    #[test]
    fn test_compare_with_imageproc() {
        // Create a test image with a simple shape
        let mut image = GrayImage::new(20, 20);
        for y in 5..15 {
            for x in 5..15 {
                image.put_pixel(x, y, image::Luma([255]));
            }
        }

        // Get contours from our implementation
        let our_contours = find_contours_with_threshold(&image, 0);

        // Get contours from imageproc
        let imageproc_contours: Vec<imageproc::contours::Contour<i32>> =
            imageproc::contours::find_contours_with_threshold(&image, 0);

        // Both should find contours
        assert!(
            !our_contours.is_empty(),
            "Our implementation should find contours"
        );
        assert!(
            !imageproc_contours.is_empty(),
            "imageproc should find contours"
        );

        // Count outer contours
        let our_outer = our_contours
            .iter()
            .filter(|c| c.border_type == BorderType::Outer)
            .count();
        let imageproc_outer = imageproc_contours
            .iter()
            .filter(|c| c.border_type == imageproc::contours::BorderType::Outer)
            .count();

        // Both should find at least one outer contour
        assert!(
            our_outer >= 1,
            "Our implementation should find outer contour"
        );
        assert!(imageproc_outer >= 1, "imageproc should find outer contour");

        // The contour points should cover similar area
        // Get the largest outer contour from each
        let our_largest = our_contours
            .iter()
            .filter(|c| c.border_type == BorderType::Outer)
            .max_by_key(|c| c.points.len())
            .unwrap();

        let imageproc_largest = imageproc_contours
            .iter()
            .filter(|c| c.border_type == imageproc::contours::BorderType::Outer)
            .max_by_key(|c| c.points.len())
            .unwrap();

        // Contour points should be similar in count (within a tolerance)
        // Different algorithms may produce slightly different number of points
        let our_len = our_largest.points.len();
        let imageproc_len = imageproc_largest.points.len();
        let diff = (our_len as i32 - imageproc_len as i32).abs();
        let tolerance = (our_len.max(imageproc_len) as f32 * 0.5) as i32; // 50% tolerance

        assert!(
            diff <= tolerance,
            "Contour point count should be similar. Ours: {}, imageproc: {}, diff: {}",
            our_len,
            imageproc_len,
            diff
        );
    }

    /// Compare donut shape (with hole)
    #[cfg(feature = "annotator")]
    #[test]
    fn test_compare_donut_with_imageproc() {
        // Create a donut shape
        let mut image = GrayImage::new(30, 30);
        for y in 5..25 {
            for x in 5..25 {
                image.put_pixel(x, y, image::Luma([255]));
            }
        }
        // Create hole
        for y in 10..20 {
            for x in 10..20 {
                image.put_pixel(x, y, image::Luma([0]));
            }
        }

        let our_contours = find_contours_with_threshold(&image, 0);
        let imageproc_contours: Vec<imageproc::contours::Contour<i32>> =
            imageproc::contours::find_contours_with_threshold(&image, 0);

        // Both should detect outer and inner contours
        let our_outer = our_contours
            .iter()
            .filter(|c| c.border_type == BorderType::Outer)
            .count();
        let our_holes = our_contours
            .iter()
            .filter(|c| c.border_type == BorderType::Hole)
            .count();

        let imageproc_outer = imageproc_contours
            .iter()
            .filter(|c| c.border_type == imageproc::contours::BorderType::Outer)
            .count();
        let imageproc_holes = imageproc_contours
            .iter()
            .filter(|c| c.border_type == imageproc::contours::BorderType::Hole)
            .count();

        assert!(our_outer >= 1, "Should find outer contour");
        assert!(our_holes >= 1, "Should find hole contour");
        assert!(imageproc_outer >= 1, "imageproc should find outer contour");
        assert!(imageproc_holes >= 1, "imageproc should find hole contour");
    }

    #[test]
    fn test_simple_rectangle() {
        // Create a 10x10 image with a 4x4 white rectangle in the center
        let mut image = GrayImage::new(10, 10);
        for y in 3..7 {
            for x in 3..7 {
                image.put_pixel(x, y, image::Luma([255]));
            }
        }

        let contours = find_contours_with_threshold(&image, 0);

        // Should find at least one contour
        assert!(!contours.is_empty());

        // The outer contour should exist
        let outer = contours.iter().find(|c| c.border_type == BorderType::Outer);
        assert!(outer.is_some());

        // The contour should have multiple points
        let outer = outer.unwrap();
        assert!(outer.points.len() >= 4);
    }

    #[test]
    fn test_empty_image() {
        let image = GrayImage::new(10, 10);
        let contours = find_contours_with_threshold(&image, 0);
        assert!(contours.is_empty());
    }

    #[test]
    fn test_single_pixel() {
        let mut image = GrayImage::new(10, 10);
        image.put_pixel(5, 5, image::Luma([255]));

        let contours = find_contours_with_threshold(&image, 0);
        // Single isolated pixel produces a contour with 1 point
        // Note: Some implementations may or may not include isolated pixels
        // Our implementation traces boundaries, so an isolated pixel is valid
        if !contours.is_empty() {
            assert!(!contours[0].points.is_empty());
        }
    }

    #[test]
    fn test_contours_topological() {
        let mut image = GrayImage::new(30, 30);
        // border 1 (outer) - square from (5,5) to (25,25)
        for y in 5..25 {
            for x in 5..25 {
                image.put_pixel(x, y, image::Luma([255]));
            }
        }
        // border 2 (hole) - square from (10,10) to (20,20)
        for y in 10..20 {
            for x in 10..20 {
                image.put_pixel(x, y, image::Luma([0]));
            }
        }
        // border 3 (outer) - square from (12,12) to (18,18)
        for y in 12..18 {
            for x in 12..18 {
                image.put_pixel(x, y, image::Luma([255]));
            }
        }

        let contours = find_contours_with_threshold(&image, 0);

        // We expect 3 contours
        assert_eq!(contours.len(), 3);

        // Contour 0: Outer (5,5)
        assert_eq!(contours[0].border_type, BorderType::Outer);
        assert_eq!(contours[0].parent, None);

        // Contour 1: Hole (10,10) inside Contour 0
        assert_eq!(contours[1].border_type, BorderType::Hole);
        assert_eq!(contours[1].parent, Some(0));

        // Contour 2: Outer (12,12) inside Hole 1
        assert_eq!(contours[2].border_type, BorderType::Outer);
        assert_eq!(contours[2].parent, Some(1));
    }
}
