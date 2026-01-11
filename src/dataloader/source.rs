use anyhow::Result;
use image::DynamicImage;
use rayon::prelude::*;
use std::path::PathBuf;
use std::str::FromStr;

use crate::SourceType;

/// Represents a collection of input sources.
#[derive(Debug, Clone, Default)]
pub struct Source {
    pub(crate) tasks: std::collections::VecDeque<SourceType>,
}

impl Source {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tasks(&self) -> &std::collections::VecDeque<SourceType> {
        &self.tasks
    }

    pub fn push(&mut self, task: SourceType) {
        if matches!(task, SourceType::Directory(_) | SourceType::Glob(_)) {
            if let Ok(tasks) = task.flatten() {
                self.tasks.extend(tasks);
            }
        } else {
            self.tasks.push_back(task);
        }
    }

    pub fn extend<I, T>(&mut self, tasks: I)
    where
        I: IntoIterator<Item = T>,
        T: Into<SourceType>,
    {
        for task in tasks {
            self.push(task.into());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn has_video_or_stream(&self) -> bool {
        self.tasks.iter().any(|t| t.is_video())
    }
}

impl From<&Source> for Source {
    fn from(s: &Source) -> Self {
        s.clone()
    }
}

impl FromStr for Source {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut source = Source::new();

        // Split by separators and trim each part
        let sep = if s.contains('|') { '|' } else { ',' };

        if s.contains(sep) {
            let parts: Vec<&str> = s
                .split(sep)
                .map(|p| p.trim())
                .filter(|p| !p.is_empty())
                .collect();

            if parts.len() > 1 {
                let tasks: Result<Vec<Vec<SourceType>>, anyhow::Error> = parts
                    .into_par_iter()
                    .map(|part| SourceType::from_str(part)?.flatten())
                    .collect();

                for t in tasks? {
                    source.tasks.extend(t);
                }
                return Ok(source);
            }
        }

        let st = SourceType::from_str(s)?;
        source.tasks.extend(st.flatten()?);

        Ok(source)
    }
}

impl<T> From<Vec<T>> for Source
where
    T: Into<SourceType>,
{
    fn from(v: Vec<T>) -> Self {
        let mut source = Source::new();
        source.extend(v);
        source
    }
}

impl<T> From<&Vec<T>> for Source
where
    T: Clone + Into<SourceType>,
{
    fn from(v: &Vec<T>) -> Self {
        let mut source = Source::new();
        source.extend(v.iter().cloned());
        source
    }
}

impl<T, const N: usize> From<[T; N]> for Source
where
    T: Into<SourceType>,
{
    fn from(v: [T; N]) -> Self {
        let mut source = Source::new();
        source.extend(v);
        source
    }
}

impl<T> From<&[T]> for Source
where
    T: Clone + Into<SourceType>,
{
    fn from(v: &[T]) -> Self {
        let mut source = Source::new();
        source.extend(v.iter().cloned());
        source
    }
}

impl From<SourceType> for Source {
    fn from(st: SourceType) -> Self {
        let mut s = Source::new();
        s.push(st);
        s
    }
}

impl From<DynamicImage> for Source {
    fn from(img: DynamicImage) -> Self {
        SourceType::from(img).into()
    }
}

impl From<u32> for Source {
    fn from(idx: u32) -> Self {
        SourceType::Webcam(idx).into()
    }
}

impl From<i32> for Source {
    fn from(idx: i32) -> Self {
        SourceType::Webcam(idx as u32).into()
    }
}

impl From<PathBuf> for Source {
    fn from(p: PathBuf) -> Self {
        SourceType::from(p).into()
    }
}

impl From<&PathBuf> for Source {
    fn from(p: &PathBuf) -> Self {
        SourceType::from(p).into()
    }
}

impl From<String> for Source {
    fn from(s: String) -> Self {
        s.as_str().parse().unwrap_or_default()
    }
}

impl From<&String> for Source {
    fn from(s: &String) -> Self {
        s.as_str().parse().unwrap_or_default()
    }
}

impl From<&str> for Source {
    fn from(s: &str) -> Self {
        s.parse().unwrap_or_default()
    }
}
