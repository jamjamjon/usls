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

