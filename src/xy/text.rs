/// Wrapper over [`String`]
#[derive(aksr::Builder, Debug, Clone, Default, PartialEq)]
pub struct Text(String);

impl std::ops::Deref for Text {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: AsRef<str>> From<T> for Text {
    fn from(x: T) -> Self {
        Self(x.as_ref().to_string())
    }
}
