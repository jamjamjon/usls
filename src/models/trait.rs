use anyhow::Result;
use usls::{elapsed, Image, Processor, Ts, Xs, Ys, X, Y};

pub trait Module: Sized {
    // Required method
    fn get_ts(&mut self) -> Option<&mut Ts>;
    fn get_processor(&mut self) -> &mut Processor;
    fn get_model(&mut self) -> Option<&mut Engine>;
    fn get_visual(&mut self) -> Option<&mut Engine>;
    fn get_textual(&mut self) -> Option<&mut Engine>;

    // Provided method
    type Input;

    fn preprocess(&self, xs: &[Self::Input]) -> Result<Xs>;

    fn inference(&mut self, xs: Xs) -> Result<Xs>;

    fn postprocess(&self, xs: Xs) -> Result<Ys>;

    fn encode_images(&mut self, xs: &Image) -> Result<X> {
        let xs = elapsed!("encode-images-preprocess", self.ts, {
            self.processor.process_images(xs)?
        });
        if let Some(model) = self.get_model() {
            let xs = elapsed!("encode-images-inference", self.ts, {
                self.visual.run(xs.into())?
            });
            let x = elapsed!("encode-images-postprocess", self.ts, { xs[0].to_owned() });
        }
    }
    fn encode_text(&mut self, xs: &Image) -> Result<X> {
        // pub fn encode_texts(&mut self, xs: &[&str]) -> Result<X> {
        //     let xs = elapsed!("textual-preprocess", self.ts, {
        //         let encodings: Vec<f32> = self
        //             .processor
        //             .encode_texts_ids(xs, false)? // skip_special_tokens
        //             .into_iter()
        //             .flatten()
        //             .collect();

        //         let x: X =
        //             Array2::from_shape_vec((xs.len(), encodings.len() / xs.len()), encodings)?
        //                 .into_dyn()
        //                 .into();

        //         x
        //     });
        //     let xs = elapsed!("textual-inference", self.ts, {
        //         self.textual.run(xs.into())?
        //     });
        //     let x = elapsed!("textual-postprocess", self.ts, { xs[0].to_owned() });

        //     Ok(x)
        // }
    }

    fn encode_texts(&mut self, xs: &[Image]) -> Result<X>;

    fn forward(&mut self, xs: &[Self::Input]) -> Result<Ys> {
        let ys = elapsed!("preprocess", self.get_ts(), { self.preprocess(xs)? });
        let ys = elapsed!("inference", self.get_ts(), { self.inference(ys)? });
        let ys = elapsed!("postprocess", self.get_ts(), { self.postprocess(ys)? });

        Ok(ys)
    }

    fn summary(&mut self) {
        self.get_ts().summary();
    }
}
