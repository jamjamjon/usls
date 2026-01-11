# Open-Set Detection Examples


## Quick Start

### grounding-dino
```
cargo run -F cuda -F vlm --example open-set-detection grounding-dino --dtype q8 --device cuda:0 --processor-device cuda:0 --kind llmdet-tiny
```

### owlv2
```
cargo run -F cuda -F vlm --example open-set-detection owlv2 --dtype fp32 --device cuda:0 --processor-device cuda:0 --kind base-ensemble
```


## Results
![](https://github.com/jamjamjon/assets/releases/download/grounding-dino/demo-llmdet-tiny-q8.jpg)