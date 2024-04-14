use crate::Polygon;

#[derive(Default, Clone, PartialEq)]
pub struct Mask {
    pub polygon: Polygon,
    pub id: usize,
    pub name: Option<String>,
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mask")
            .field("polygons(num_points)", &self.polygon.points.len())
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

impl Mask {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}
