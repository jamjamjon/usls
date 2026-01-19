# Matting Examples

This directory contains examples for image matting and segmentation models.

## Examples

### MediaPipe Selfie Segmentation with Background Removal
```bash
cargo run -F cuda-full --example image-matting -- mediapipe  --device cuda --processor-device cuda
```

### MODNet Portrait Matting
```bash
cargo run -F cuda-full --example image-matting -- modnet  --device cuda --processor-device cuda
```

### BiRefNet Models
```bash
 cargo run -F cuda-12040 -F ort-load-dynamic --example image-matting -- birefnet --device cuda:2 --processor-device cuda:2 --variant matting
```