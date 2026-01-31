# Results

All models return results in a unified `Y` structure.

!!! tip "Quick Start"
    ```rust
    let ys: Vec<Y> = model.run(&images)?;
    ```

---

## Y Structure

| Field | Type | Description |
| :--- | :--- | :--- |
| `hbbs` | `Vec<Hbb>` | Horizontal bounding boxes |
| `obbs` | `Vec<Obb>` | Oriented bounding boxes |
| `masks` | `Vec<Mask>` | Segmentation masks |
| `polygons` | `Vec<Polygon>` | Contours |
| `keypoints` | `Vec<Keypoint>` | Keypoints |
| `keypointss` | `Vec<Vec<Keypoint>>` | Multiple keypoint sets |
| `probs` | `Vec<Prob>` | Classification probabilities |
| `texts` | `Vec<Text>` | OCR/VLM text |
| `embedding` | `X` | Feature embeddings |
| `extra` | `HashMap<String, X>` | Model-specific data |
| `images` | `Vec<Image>` | Output images |

---

## Common Methods

All result types have metadata methods via `impl_meta_methods!`:

| Method | Returns | Description |
| :--- | :--- | :--- |
| `id()` | `Option<usize>` | Class ID |
| `name()` | `Option<&str>` | Class name |
| `confidence()` | `Option<f32>` | Confidence score |
| `track_id()` | `Option<usize>` | Tracking ID |
| `uid()` | `usize` | Unique instance ID |

---

## Hbb (Horizontal Bounding Box)

### Geometry Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `xmin()` / `ymin()` | `f32` | Top-left corner |
| `xmax()` / `ymax()` | `f32` | Bottom-right corner |
| `cx()` / `cy()` | `f32` | Center coordinates |
| `width()` / `height()` | `f32` | Box dimensions |
| `xyxy()` | `(f32, f32, f32, f32)` | [x1, y1, x2, y2] |
| `xywh()` | `(f32, f32, f32, f32)` | [x, y, w, h] |
| `cxywh()` | `(f32, f32, f32, f32)` | [cx, cy, w, h] |
| `cxcyah()` | `(f32, f32, f32, f32)` | [cx, cy, ar, h] |
| `area()` | `f32` | Box area |
| `perimeter()` | `f32` | Box perimeter |
| `is_square()` | `bool` | Check if square |

### Operations

| Method | Description |
| :--- | :--- |
| `intersect(&other)` | Intersection area |
| `union(&other)` | Union area |
| `iou(&other)` | IoU ratio |
| `contains(&other)` | Contains check |
| `to_polygon()` | Convert to Polygon |

### Constructors

| Method | Description |
| :--- | :--- |
| `from_xyxy(x1, y1, x2, y2)` | From corners |
| `from_xywh(x, y, w, h)` | From position + size |
| `from_cxcywh(cx, cy, w, h)` | From center + size |

!!! example "Example"
    ```rust
    for hbb in y.hbbs() {
        let [x1, y1, x2, y2] = [hbb.xmin(), hbb.ymin(), hbb.xmax(), hbb.ymax()];
        println!("{}: {:.2}%", hbb.name().unwrap_or("?"), hbb.confidence().unwrap_or(0.0) * 100.0);
    }
    ```

---

## Obb (Oriented Bounding Box)

### Geometry Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `coords()` | `&[[f32; 2]; 4]` | 4 vertices (CCW) |
| `area()` | `f32` | Polygon area |
| `top()` / `left()` | `[f32; 2]` | Extreme points |
| `bottom()` / `right()` | `[f32; 2]` | Extreme points |
| `is_hbb()` | `bool` | Check axis-aligned |

### Operations

| Method | Description |
| :--- | :--- |
| `intersect(&other)` | Intersection area (Sutherland-Hodgman) |
| `union(&other)` | Union area |
| `iou(&other)` | IoU ratio |
| `to_polygon()` | Convert to Polygon |

### Constructors

| Method | Description |
| :--- | :--- |
| `from_cxcywhd(cx, cy, w, h, degrees)` | Center + size + angle (deg) |
| `from_cxcywhr(cx, cy, w, h, radians)` | Center + size + angle (rad) |

---

## Mask

| Method | Returns | Description |
| :--- | :--- | :--- |
| `width()` / `height()` | `u32` | Dimensions |
| `dimensions()` | `(u32, u32)` | (w, h) |
| `to_vec()` | `Vec<u8>` | Raw data |
| `polygon()` | `Option<Polygon>` | Largest contour |
| `polygons()` | `Vec<Polygon>` | All contours |

!!! example "Example"
    ```rust
    for mask in y.masks() {
        println!("Mask: {}x{}", mask.width(), mask.height());
        if let Some(poly) = mask.polygon() {
            println!("Area: {}", poly.area());
        }
    }
    ```

---

## Polygon

| Method | Returns | Description |
| :--- | :--- | :--- |
| `count()` | `usize` | Number of points |
| `points()` | `Vec<[f32; 2]>` | Clone of coords |
| `exterior()` | `&[[f32; 2]]` | Coords slice |
| `is_closed()` | `bool` | Check if closed |
| `area()` | `f64` | Shoelace formula |
| `perimeter()` | `f64` | Euclidean length |
| `centroid()` | `Option<(f32, f32)>` | Center point |

### Operations

| Method | Description |
| :--- | :--- |
| `intersect(&other)` | Intersection area |
| `union(&other)` | Union area |
| `hbb()` | Bounding box |
| `obb()` | Minimum rotated rect |
| `convex_hull()` | Convex hull |
| `simplify(eps)` | RDP simplification |
| `resample(n)` | Add points on edges |
| `unclip(delta, w, h)` | Expand polygon |

---

## Keypoint

| Method | Returns | Description |
| :--- | :--- | :--- |
| `x()` / `y()` | `f32` | Coordinates |
| `xy()` | `(f32, f32)` | (x, y) tuple |
| `is_origin()` | `bool` | Check (0, 0) |
| `distance_from(&other)` | `f32` | Euclidean distance |
| `distance_from_origin()` | `f32` | Distance from (0,0) |
| `rotate(rad)` | `Self` | Rotate around origin |
| `cross(&other)` | `f32` | Cross product |

### Operators

`+`, `-`, `*`, `/` with `f32` or `Keypoint`

---

## Prob (Classification)

| Method | Description |
| :--- | :--- |
| `new_probs(probs, names, k)` | Create top-k probs |

---

## Text (OCR/VLM)

| Method | Returns | Description |
| :--- | :--- | :--- |
| `text()` | `&str` | Text content |

---

## Access Patterns

### Borrow (Reference)

!!! example "Example"
    ```rust
    let ys = model.run(&xs)?;

    for y in &ys {
        for hbb in y.hbbs() {
            println!("Box: {:?}, Conf: {:?}", hbb.xyxy(), hbb.confidence());
        }
    }
    ```

### Consume (Ownership)

!!! example "Example"
    ```rust
    let ys = model.run(&xs)?;

    for y in ys {
        let boxes: Vec<Hbb> = y.hbbs;
    }
    ```
