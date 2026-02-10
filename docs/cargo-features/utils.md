# Utilities

| Feature | Category | Description | Dependencies | Default |
|---------|----------|-------------|-------------|:-------:|
| ***`annotator`*** | Annotation | Draw bounding boxes, keypoints, masks on images | `ab_glyph`, `imageproc` | ✓ |
| **`viewer`** | Visualization | Real-time image/video display (like OpenCV `imshow`) | `minifb` | x |
| **`video`** | I/O | Video read/write streaming support | `video-rs` | x |

!!! tip "Usage Example"
    ```toml
    # Default: annotation only
    usls = "0.1"
    
    # With viewer and video
    usls = { version = "0.1", features = ["viewer", "video"] }
    ```
