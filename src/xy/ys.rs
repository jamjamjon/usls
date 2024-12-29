use crate::Y;

/// Wrapper over `Vec<Y>`
#[derive(aksr::Builder, Default, Debug)]
pub struct Ys(pub Vec<Y>);

impl From<Vec<Y>> for Ys {
    fn from(xs: Vec<Y>) -> Self {
        Self(xs)
    }
}

impl std::ops::Deref for Ys {
    type Target = Vec<Y>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
