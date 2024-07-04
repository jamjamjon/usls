## v0.0.4 - 2024-06-30

### Added

- Add X struct to handle input and preprocessing
- Add Ops struct to manage common operations
- Use SIMD (fast_image_resize) to accelerate model pre-processing and post-processing.YOLOv8-seg post-processing (~120ms => ~20ms), Depth-Anything post-processing (~23ms => ~2ms).

### Deprecated

- Mark `Ops::descale_mask()` as deprecated.

### Fixed

### Changed

### Removed

### Refactored

### Others
