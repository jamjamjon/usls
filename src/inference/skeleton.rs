use crate::Color;

#[derive(Debug, Clone, Default)]
pub struct Connection {
    pub indices: (usize, usize),
    pub color: Option<Color>,
}

impl From<(usize, usize)> for Connection {
    fn from(indices: (usize, usize)) -> Self {
        Self {
            indices,
            color: None,
        }
    }
}

impl From<(usize, usize, Color)> for Connection {
    fn from((a, b, color): (usize, usize, Color)) -> Self {
        Self {
            indices: (a, b),
            color: Some(color),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Skeleton {
    pub connections: Vec<Connection>,
}

impl std::ops::Deref for Skeleton {
    type Target = Vec<Connection>;

    fn deref(&self) -> &Self::Target {
        &self.connections
    }
}

impl Skeleton {
    pub fn with_connections<C: Into<Connection> + Clone>(mut self, connections: &[C]) -> Self {
        self.connections = connections.iter().cloned().map(|c| c.into()).collect();
        self
    }

    pub fn with_colors(mut self, colors: &[Color]) -> Self {
        for (i, connection) in self.connections.iter_mut().enumerate() {
            if i < colors.len() {
                connection.color = Some(colors[i]);
            }
        }
        self
    }
}

impl From<&[(usize, usize)]> for Skeleton {
    fn from(connections: &[(usize, usize)]) -> Self {
        Self {
            connections: connections.iter().map(|&c| c.into()).collect(),
        }
    }
}

impl<const N: usize> From<[(usize, usize); N]> for Skeleton {
    fn from(arr: [(usize, usize); N]) -> Self {
        Self::from(arr.as_slice())
    }
}

impl From<(&[(usize, usize)], &[Color])> for Skeleton {
    fn from((connections, colors): (&[(usize, usize)], &[Color])) -> Self {
        Self {
            connections: connections
                .iter()
                .zip(colors.iter())
                .map(|(&(a, b), &c)| (a, b, c).into())
                .collect(),
        }
    }
}

impl<const N: usize> From<([(usize, usize); N], [Color; N])> for Skeleton {
    fn from((connections, colors): ([(usize, usize); N], [Color; N])) -> Self {
        Skeleton::from((&connections[..], &colors[..]))
    }
}

pub const COCO_SKELETON_16: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

// pub const COCO_SKELETON_COLOR_16: [(usize, usize); 16] = [];
