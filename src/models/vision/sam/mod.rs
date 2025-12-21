mod config;
mod r#impl;

pub use r#impl::*;

/// SAM prompt containing coordinates and labels for segmentation.
#[derive(Debug, Default, Clone)]
pub struct SamPrompt {
    /// Point coordinates for prompting.
    pub coords: Vec<Vec<[f32; 2]>>,
    /// Labels corresponding to the coordinates.
    pub labels: Vec<Vec<f32>>,
}

impl SamPrompt {
    pub fn point_coords(&self, ratio: f32) -> anyhow::Result<crate::X> {
        // [num_labels,num_points,2]
        let num_labels = self.coords.len();
        let num_points = if num_labels > 0 {
            self.coords[0].len()
        } else {
            0
        };
        let flat: Vec<f32> = self
            .coords
            .iter()
            .flat_map(|v| v.iter().flat_map(|&[x, y]| [x, y]))
            .collect();
        let y = ndarray::Array3::from_shape_vec((num_labels, num_points, 2), flat)?.into_dyn();

        Ok((y * ratio).into())
    }

    pub fn point_labels(&self) -> anyhow::Result<crate::X> {
        // [num_labels,num_points]
        let num_labels = self.labels.len();
        let num_points = if num_labels > 0 {
            self.labels[0].len()
        } else {
            0
        };
        let flat: Vec<f32> = self.labels.iter().flat_map(|v| v.iter().copied()).collect();
        let y = ndarray::Array2::from_shape_vec((num_labels, num_points), flat)?.into_dyn();
        Ok(y.into())
    }

    pub fn with_xyxy(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        // TODO: if already has points, push_front coords
        self.coords.push(vec![[x1, y1], [x2, y2]]);
        self.labels.push(vec![2., 3.]);

        self
    }

    pub fn with_positive_point(mut self, x: f32, y: f32) -> Self {
        self = self.add_point(x, y, 1.);
        self
    }

    pub fn with_negative_point(mut self, x: f32, y: f32) -> Self {
        self = self.add_point(x, y, 0.);
        self
    }

    fn add_point(mut self, x: f32, y: f32, id: f32) -> Self {
        if self.coords.is_empty() {
            self.coords.push(vec![[x, y]]);
            self.labels.push(vec![id]);
        } else {
            if let Some(last) = self.coords.last_mut() {
                last.extend_from_slice(&[[x, y]]);
            }

            if let Some(last) = self.labels.last_mut() {
                last.extend_from_slice(&[id]);
            }
        }
        self
    }

    pub fn with_positive_point_object(mut self, x: f32, y: f32) -> Self {
        self.coords.push(vec![[x, y]]);
        self.labels.push(vec![1.]);
        self
    }

    pub fn with_negative_point_object(mut self, x: f32, y: f32) -> Self {
        self.coords.push(vec![[x, y]]);
        self.labels.push(vec![0.]);
        self
    }
}
