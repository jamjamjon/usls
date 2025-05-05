mod basemodel;
mod config;
mod image_classifier;

pub use basemodel::*;
pub use config::*;
pub use image_classifier::*;

// --------------------------------------
use crate::{elapsed, Image, Ts, Xs, X, Y};
use anyhow::Result;
pub trait Model: Sized {
    fn get_ts_mut(&mut self) -> &mut Ts;

    fn summary(&mut self) {
        self.get_ts_mut().summary();
    }

    // fn summary(&self);
    // fn spec(&self) -> &str;
    // fn batch(&self) -> usize;
}

pub trait VisualModel: Model {
    fn preprocess(&mut self, xs: &[Image]) -> Result<Xs>;
    fn inference(&mut self, xs: Xs) -> Result<Xs>;
    fn postprocess(&mut self, xs: Xs) -> Result<Vec<Y>>;

    fn forward(&mut self, xs: &[Image]) -> Result<Vec<Y>> {
        let ys = elapsed!("preprocess", self.get_ts_mut(), { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.get_ts_mut(), { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.get_ts_mut(), { self.postprocess(ys)? });

        Ok(ys)
    }

    fn encode_images(&mut self, xs: &[Image]) -> Result<X>;
}

// pub trait MultimodalModel: Model {
//     fn encode_images(&mut self, xs: &[Image]) -> Result<X>;
//     fn encode_texts(&mut self, xs: &[&str]) -> Result<X>;
// }
