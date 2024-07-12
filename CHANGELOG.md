## v0.0.5 - 2024-07-12

### Changed

- Accelerated `YOLO`'s post-processing using `Rayon`. Now, `YOLOv8-seg` takes only around **8ms**, depending on your machine. Note that this repo's implementation of YOLO-Seg saves not only the masks but also their contour points. The official YOLOv8 Python version only saves the masks, making it appear much faster.
- Merged all YOLOv8-related solution models into YOLO examples.
- Consolidated all YOLO-series model examples into the YOLO example.
- Refactored the `YOLO` struct to unify all `YOLO versions` and `YOLO tasks`. It now supports user-defined YOLO models with different `Preds Tensor Formats`.
- Introduced a new `Nms` trait, combining `apply_bboxes_nms()` and `apply_mbrs_nms()` into `apply_nms()`.

### Added

- Added support for `YOLOv6` and `YOLOv7`.
- Updated documentation for `y.rs`.
- Updated documentation for `bbox.rs`.
- Updated the README.md.
- Added `with_yolo_preds()` to `Options`.
- Added support for Depth-Anything-v2.
- Added `RTDETR` to the `YOLOVersion` struct.

### Removed

- Merged the following models' examples into the YOLOv8 example: `yolov8-face`, `yolov8-falldown`, `yolov8-head`, `yolov8-trash`, `fastsam`, and `face-parsing`.
- Removed `anchors_first`, `conf_independent`, and their related methods from `Options`.


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
