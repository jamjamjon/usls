# Open-Set Detection Examples


## Quick Start

### grounding-dino
```bash
# Single prompt
cargo run -F cuda-full -F vlm --example open-set-detection grounding-dino --dtype fp16 --device cuda:0 --processor-device cuda:0 --kind llmdet-tiny

# Multiple prompts
cargo run -F cuda-full -F vlm --example open-set-detection grounding-dino --dtype fp16 --device cuda:0 --processor-device cuda:0 --kind llmdet-tiny -p person -p bus -p dog
```

### owlv2
```bash
# Single prompt
cargo run -F cuda-full -F vlm --example open-set-detection owlv2 --dtype fp32 --device cuda:0 --processor-device cuda:0 --kind base-ensemble -p shoe 
```


## Results
![](https://github.com/jamjamjon/assets/releases/download/grounding-dino/demo-llmdet-tiny-q8.jpg)