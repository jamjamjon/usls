use crate::{Options, Xs, Y};

pub trait Vision: Sized {
    type Input; // DynamicImage

    /// Creates a new instance of the model with the given options.
    fn new(options: Options) -> anyhow::Result<Self>;

    /// Preprocesses the input data.
    fn preprocess(&self, xs: &[Self::Input]) -> anyhow::Result<Xs>;

    /// Executes the model on the preprocessed data.
    fn inference(&mut self, xs: Xs) -> anyhow::Result<Xs>;

    /// Postprocesses the model's output.
    fn postprocess(&self, xs: Xs, xs0: &[Self::Input]) -> anyhow::Result<Vec<Y>>;

    /// Executes the full pipeline.
    fn run(&mut self, xs: &[Self::Input]) -> anyhow::Result<Vec<Y>> {
        let ys = self.preprocess(xs)?;
        let ys = self.inference(ys)?;
        let ys = self.postprocess(ys, xs)?;
        Ok(ys)
    }

    /// Executes the full pipeline.
    fn forward(&mut self, xs: &[Self::Input], profile: bool) -> anyhow::Result<Vec<Y>> {
        let span = tracing::span!(tracing::Level::INFO, "DataLoader-new");
        let _guard = span.enter();

        let t_pre = std::time::Instant::now();
        let ys = self.preprocess(xs)?;
        let t_pre = t_pre.elapsed();

        let t_exe = std::time::Instant::now();
        let ys = self.inference(ys)?;
        let t_exe = t_exe.elapsed();

        let t_post = std::time::Instant::now();
        let ys = self.postprocess(ys, xs)?;
        let t_post = t_post.elapsed();

        if profile {
            tracing::info!(
                "> Preprocess: {t_pre:?} | Execution: {t_exe:?} | Postprocess: {t_post:?}"
            );
        }

        Ok(ys)
    }
}
