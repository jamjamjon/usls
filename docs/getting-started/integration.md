# Integration Workflow

`usls` implements a clean, modular pipeline from data ingestion to results visualization.

## The 4-Step Pipeline

1.  **Configure Model**: Select a pre-configured model (e.g., `Config::rfdetr_nano()`), customize settings, and commit the configuration.
2.  **Load Data**: Setup a `DataLoader` to handle your input sources (images, videos, etc.).
3.  **Inference**: Iterate through the `DataLoader` and pass data to `model.run()` or `model.forward()`.
4.  **Extract Results**: Access detections, masks, or embeddings from the unified `Y` output.
5.  **Annotate (Optional)**: Use the `Annotator` to draw results back onto the original images.
6.  **Visualize (Optional)**: Use the `Viewer` for real-time display or video recording.


!!! tip "Example"
    ```rust
    use usls::*;

    fn main() -> anyhow::Result<()> {
        // 1. Configure & Build Model
        let config = Config::rfdetr_nano()
            .with_model_device(Device::Cuda(0))
            .commit()?;
        let mut model = RFDETR::new(config)?;
        
        // 2. Setup DataLoader
        let dl = DataLoader::new("image.jpg")?
            .with_batch(model.batch())
            .stream()?;
        
        // optional: Annotate
        let annotator = Annotator::default();

        // optional: Viewer
        let mut viewer = Viewer::default();

        // 3. Run Inference
        for xs in dl {
            let ys = model.run(&xs)?;
            for (x, y) in xs.iter().zip(ys.iter()) {
                // 4. Access results
                for hb in y.hbbs() {
                    println!("{}", hb);
                }

                // optional: Check if the window is closed and exit if so.
                if viewer.is_window_exist_and_closed() {
                    break;
                }

                // optional: Annotate
                let image_annotated = annotator.annotate(x, y)?;

                // optional: Display the current image.
                viewer.imshow(&image_annotated)?;

                // optional: Wait for a key press or timeout, and exit on Escape.
                if let Some(key) = viewer.wait_key(10) {
                    if key == usls::Key::Escape {
                        break;
                    }
                }

                // optional: Save the annotated image.
                image_annotated.save("output.jpg")?;
            }
        }
        Ok(())
    }
    ```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open-page-variant:{ .lg .middle } **Guides**

    ---

    Learn more about modules, config, and advanced usage

    [Read Guides →](../guides/overview.md)

-   :material-package-variant:{ .lg .middle } **Model Zoo**

    ---

    Explore 50+ pre-trained models

    [Browse Models →](../model-zoo/overview.md)

-   :material-help-circle:{ .lg .middle } **FAQ**

    ---

    Find answers to common questions

    [View FAQ →](../faq.md)

</div>
