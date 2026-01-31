# Viewer

The `Viewer` provides real-time image/video display with keyboard interaction and video recording capabilities.



## Window Management

| Method | Description |
| :--- | :--- |
| `is_window_exist_and_closed()` | Check if user closed window |
| `wait_key(ms)` | Wait for key (timeout in ms) |
| `is_open()` | Check if window is open |

!!! warning ""
    Always check `is_window_exist_and_closed()` in your loop.


---

## Key Handling

!!! tip "Common Keys"
    - `ESC` — Exit
    - `Space` — Pause/Resume
    - `S` — Save screenshot

!!! example "Example"
    ```rust
    if let Some(key) = viewer.wait_key(30) {
        match key {
            Key::Escape => break,
            Key::S => {
                images[0].save("screenshot.png")?;
            }
            _ => {}
        }
    }
    ```

---

## Video Recording

!!! note "Requires `video` feature"
    Add to `Cargo.toml`: `features = ["video"]`

!!! example "Example"
    ```rust
    let mut viewer = Viewer::default()
        .with_record_output("output.mp4")?;

    for (images, _) in dl {
        viewer.imshow(&images[0])?;
        viewer.write_video_frame(&images[0])?;
        
        if viewer.is_window_exist_and_closed() {
            break;
        }
    }
    ```

---

## Example

!!! success "imshow"
    ```rust
    use clap::Parser;
    use usls::{DataLoader, Source, Viewer};

    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Args {
        /// Data source.
        #[arg(long, required = true)]
        source: Source,

        /// Save frames to video.
        #[arg(long, default_value = "false")]
        save: bool,

        /// Window scale.
        #[arg(long, default_value = "0.8")]
        window_scale: f32,

        /// Delay in milliseconds between frames.
        #[arg(long, default_value = "1")]
        delay: u64,

        /// Num of frames to skip.
        #[arg(long, default_value = "0")]
        nfv_skip: u64,
    }

    fn main() -> anyhow::Result<()> {
        let args = Args::parse();
        let dl = DataLoader::new(args.source)?
            .with_nfv_skip(args.nfv_skip)
            .stream()?;
        let mut viewer = Viewer::default().with_window_scale(args.window_scale);

        for images in &dl {
            // Check if the window is closed and exit if so.
            if viewer.is_window_exist_and_closed() {
                break;
            }

            // Display the current image.
            viewer.imshow(&images[0])?;

            // Wait for a key press or timeout, and exit on Escape.
            if let Some(key) = viewer.wait_key(args.delay) {
                if key == usls::Key::Escape {
                    break;
                }
            }

            // Save the current frame to video if requested.
            // Note: For multiple videos, frames will be saved to separate files.
            if args.save {
                viewer.write_video_frame(&images[0])?;
            }
        }

        Ok(())
    }

    ```

---
