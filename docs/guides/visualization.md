# Visualization (Viewer)

The `Viewer` utility (powered by `minifb`) provides a simple way to display images, videos, and model results in a real-time window.

## Features

- **Cross-Platform**: Works on Linux, macOS, and Windows.
- **Window Management**: Easy creation and scaling of display windows.
- **Key Event Handling**: Capture keyboard input for interactive applications.
- **Video Recording**: Record the displayed frames to a video file.

## Basic Usage

```rust
use usls::*;

let mut viewer = Viewer::default();

for xs in &dl {
    // Check if window was closed by user
    if viewer.is_window_exist_and_closed() {
        break;
    }

    // Display the first image in batch
    viewer.imshow(&xs[0])?;
    
    // Wait for key (30ms)
    if let Some(key) = viewer.wait_key(30) {
        if key == Key::Escape {
            break;
        }
    }
}
```

## Window Scaling

Adjust the window size without changing the underlying image resolution:

***TODO***


## Recording Video

***TODO***

---

*For drawing shapes on images, see the [Annotator](annotator.md) guide.*
