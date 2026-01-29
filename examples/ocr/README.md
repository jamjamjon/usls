# OCR Examples

This directory contains examples for OCR (Optical Character Recognition) models.


### Text detection

```bash
# DB (ppocr det)
cargo run -F cuda-full -F vlm --example ocr -- db --device cuda:0 --processor-device cuda:0

# FAST / LinkNet
cargo run -F cuda-full -F vlm --example ocr -- fast --device cuda:0 --processor-device cuda:0
cargo run -F cuda-full -F vlm --example ocr -- linknet --device cuda:0 --processor-device cuda:0
```

### Text recognition

```bash
# SVTR
cargo run -F cuda-full -F vlm --example ocr -- svtr --device cuda:0 --processor-device cuda:0 --source ./examples/ocr/images-rec

# TrOCR (module-specific device/dtype)
cargo run -r -F cuda-full -F vlm --example ocr -- trocr --visual-dtype fp16 --visual-device cuda:0 --textual-decoder-dtype fp16 --textual-decoder-device cuda:0 --processor-device cuda:0 --scale s --kind printed --source ./examples/ocr/images-rec
```

### Document layout detection

```bash

## DocLayout-YOLO
cargo run -F cuda-full -F vlm --example ocr -- doclayout-yolo --device cuda:0 --processor-device cuda:0 --source images/academic.jpg

## PicoDet-Layout
cargo run -F cuda-full -F vlm --example ocr -- picodet-layout --device cuda:0 --processor-device cuda:0 --source images/academic.jpg

## PP-DocLayout v1/v2
cargo run -F cuda-full -F vlm --example ocr -- pp-doclayout --device cuda:0 --processor-device cuda:0 --source images/academic.jpg --ver 1 --dtype fp32
```

### Table structure recognition

```bash
# SLANet
cargo run -F cuda-full -F vlm --example ocr -- slanet --device cuda:0 --processor-device cuda:0 --source ./examples/ocr/images-det/table.png
```

## Results

![](https://github.com/jamjamjon/assets/releases/download/db/demo-paper.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-slanted-text-number.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-table-en.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-table-ch.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-sign.png)
![](https://github.com/jamjamjon/assets/releases/download/yolo/demo-doclayout-yolo.png)
