use anyhow::Result;
use std::sync::mpsc;

use crate::{DataLoader, Image, PB};

trait DataLoaderIterator {
    type Receiver;

    fn receiver(&self) -> &Self::Receiver;
    fn progress_bar(&self) -> Option<&PB>;
    fn next_impl(
        &mut self,
        recv_result: Result<Vec<Image>, mpsc::RecvError>,
    ) -> Option<Vec<Image>> {
        match self.progress_bar() {
            Some(progress_bar) => match recv_result {
                Ok(item) => {
                    if let Some(first) = item.first() {
                        if let Some(source) = &first.source {
                            if let Some(name) = source.file_name().and_then(|n| n.to_str()) {
                                progress_bar.set_message(name);
                            }
                        }
                    }
                    progress_bar.inc(item.len() as u64);
                    Some(item)
                }
                Err(_) => {
                    progress_bar.finish(Some("Image/Frame"));
                    None
                }
            },
            None => recv_result.ok(),
        }
    }
}

/// An iterator implementation for `DataLoader` that enables batch processing of images.
///
/// This struct is created by the `into_iter` method on `DataLoader`.
/// It provides functionality for:
/// - Receiving batches of images through a channel
/// - Tracking progress with an optional progress bar
/// - Processing images in configurable batch sizes
pub struct DataLoaderIntoIterator {
    /// Channel receiver for getting batches of images
    pub(crate) receiver: mpsc::Receiver<Vec<Image>>,
    /// Optional progress bar for tracking iteration progress
    pub(crate) progress_bar: Option<PB>,
}

impl DataLoaderIterator for DataLoaderIntoIterator {
    type Receiver = mpsc::Receiver<Vec<Image>>;

    fn receiver(&self) -> &Self::Receiver {
        &self.receiver
    }

    fn progress_bar(&self) -> Option<&PB> {
        self.progress_bar.as_ref()
    }
}

impl Iterator for DataLoaderIntoIterator {
    type Item = Vec<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl(self.receiver().recv())
    }
}

impl IntoIterator for DataLoader {
    type Item = Vec<Image>;
    type IntoIter = DataLoaderIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        DataLoaderIntoIterator {
            receiver: self.receiver,
            progress_bar: self.progress_bar,
        }
    }
}

/// A borrowing iterator for `DataLoader` that enables batch processing of images.
///
/// This iterator is created by the `iter()` method on `DataLoader`, allowing iteration
/// over batches of images without taking ownership of the `DataLoader`.
///
/// # Fields
/// - `receiver`: A reference to the channel receiver that provides batches of images
/// - `progress_bar`: An optional reference to a progress bar for tracking iteration progress
pub struct DataLoaderIter<'a> {
    pub(crate) receiver: &'a mpsc::Receiver<Vec<Image>>,
    pub(crate) progress_bar: Option<&'a PB>,
}

impl DataLoaderIterator for DataLoaderIter<'_> {
    type Receiver = mpsc::Receiver<Vec<Image>>;

    fn receiver(&self) -> &Self::Receiver {
        self.receiver
    }

    fn progress_bar(&self) -> Option<&PB> {
        self.progress_bar
    }
}

impl Iterator for DataLoaderIter<'_> {
    type Item = Vec<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_impl(self.receiver().recv())
    }
}

impl<'a> IntoIterator for &'a DataLoader {
    type Item = Vec<Image>;
    type IntoIter = DataLoaderIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DataLoaderIter {
            receiver: &self.receiver,
            progress_bar: self.progress_bar.as_ref(),
        }
    }
}
