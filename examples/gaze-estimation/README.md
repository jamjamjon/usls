# Gaze Estimation Examples

## Quick Start

### mobile_gaze
```
cargo run -F cuda-full,ort-load-dynamic,video --example gaze-estimation -- mobile-gaze --device cuda:0 --processor-device cuda:0 --batch 4 --source /path/to/video
```



### Results

![mobile_gaze](https://github.com/jamjamjon/assets/releases/download/mobile_gaze/demo.jpg)
