# Image Formats

| Feature | Format | Description | Default |
|---------|--------|-------------|:-------:|
| `image-gif` | GIF | Graphics Interchange Format | x |
| `image-bmp` | BMP | Windows Bitmap | x |
| `image-ico` | ICO | Windows Icon | x |
| `image-avif` | AVIF | AV1 Image File Format | x |
| `image-tiff` | TIFF | Tagged Image File Format | x |
| `image-dds` | DDS | DirectDraw Surface | x |
| `image-exr` | OpenEXR | High dynamic-range imaging | x |
| `image-ff` | Farbfeld | Simple lossless format | x |
| `image-hdr` | Radiance HDR |  | x |
| `image-pnm` | PNM | Portable Anymap (PPM/PGM/PBM) | x |
| `image-qoi` | QOI | Quite OK Image format | x |
| `image-tga` | TGA | Truevision Targa | x |
| `image-all-formats` | - | Enable all optional formats at once | x |


!!! info "Default Support"
    `jpeg`, `png`, `webp`, `rayon` are enabled by default (via `image` crate).

!!! tip "Usage Example"
    ```toml
    # Enable optional formats
    usls = { version = "0.1", features = ["image-avif", "image-tiff"] }
    ```
