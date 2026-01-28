# Result Types (Y)

All inference results in **usls** are returned in a unified structure called `Y`. This abstraction allows you to handle results from different models (detection, segmentation, classification) using a consistent API.

## 📊 The `Y` Struct

The `Y` struct acts as a container for various result types. Most models will populate only one or two fields, while complex models (like YOLOv8-seg) might populate several.

```rust
pub struct Y {
    pub hbbs: Vec<Hbb>,             // Horizontal Bounding Boxes
    pub obbs: Vec<Obb>,             // Oriented Bounding Boxes
    pub masks: Vec<Mask>,           // Segmentation Masks
    pub polygons: Vec<Polygon>,     // Contours/Polygons
    pub keypoints: Vec<Keypoint>,   // Single Keypoint sets
    pub keypointss: Vec<Vec<Keypoint>>, // Multiple Keypoint sets (Batched)
    pub probs: Vec<Prob>,           // Classification Probabilities
    pub texts: Vec<Text>,           // Text outputs (OCR/VLM)
    pub embedding: X,               // Feature Embeddings
    pub extras: HashMap<String, X>, // Model-specific custom outputs
}
```

## Accessing Results

### Borrowing (Preferred)
Use the accessor methods to borrow data without taking ownership.

```rust
let ys = model.run(&xs)?;
for y in &ys {
    // Borrow horizontal bounding boxes
    for hbb in y.hbbs() {
        println!("Box: {:?}, Conf: {}", hbb.rect(), hbb.conf());
    }
}
```

### Consuming (Ownership)
Access the fields directly if you need to move the data.

```rust
let ys = model.run(&xs)?;
for y in ys {
    let boxes = y.hbbs; // Moves Vec<Hbb> out of y
}
```

## Result Item Details

### Bounding Boxes (Hbb / Obb)
- `rect()`: returns `[x1, y1, x2, y2]` or oriented corners.
- `conf()`: detection confidence.
- `id()`: class index.
- `name()`: class label.

### Masks
- Typically high-resolution binary or grayscale masks matching the input image size.

### Keypoints
- Used for human pose estimation or landmark detection. Contains `[x, y, confidence]`.

---

*Next: [Annotator](annotator.md) - Learn how to draw these results.*
