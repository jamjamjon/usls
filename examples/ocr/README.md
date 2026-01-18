# OCR Examples

This directory contains examples for various OCR (Optical Character Recognition) models.

## Examples

### Text Detection with DB
```bash
cargo run -F cuda-full -F vlm --example ocr -- db --device cuda:0 --processor-device cuda:0
```

### Text Detection with Fast
```bash
cargo run -F cuda-full -F vlm --example ocr -- fast --device cuda:0 --processor-device cuda:0
```

### Text Detection with Linknet
```bash
cargo run -F cuda-full -F vlm --example ocr -- linknet --device cuda:0 --processor-device cuda:0
```

### Text Recognition with SVTR
```bash
cargo run -F cuda-full -F vlm --example ocr -- svtr --device cuda:0 --processor-device cuda:0 --source ./examples/ocr/images-rec
```


### Text Recognition: TrOCR
```bash
# Using module-specific device/dtype for visual encoder and textual decoder
cargo run -r -F cuda-full -F vlm --example ocr -- trocr --visual-dtype fp16 --visual-device cuda:0 --textual-decoder-dtype fp16 --textual-decoder-device cuda:0 --processor-device cuda:0 --scale s --kind printed --source ./examples/ocr/images-rec
```

### Document Layout detection: Doclayout-yolo
```bash
cargo run -F cuda-full -F vlm --example ocr -- doclayout-yolo --device cuda:0 --processor-device cuda:0 --source images/academic.jpg
```

### Document Layout detection: PicoDet-Layout
```bash
cargo run -F cuda-full -F vlm --example ocr -- picodet-layout --device cuda:0 --processor-device cuda:0 --source images/academic.jpg
```

### Table Structure Recognition: Slanet
```bash
cargo run -F cuda-full -F vlm --example ocr -- slanet --device cuda:0 --processor-device cuda:0 --source ./examples/ocr/images-det/table.png

```


## Results

![](https://github.com/jamjamjon/assets/releases/download/db/demo-paper.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-slanted-text-number.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-table-en.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-table-ch.png)
![](https://github.com/jamjamjon/assets/releases/download/db/demo-sign.png)
![](https://github.com/jamjamjon/assets/releases/download/yolo/demo-doclayout-yolo.png)
