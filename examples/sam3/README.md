
## Usage

```bash
# Basic text prompt
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype q4f16 --source assets/sam3-demo.jpg \
    -p shoe 

# Multiple prompts
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype fp16 --source assets/sam3-demo.jpg \
    -p "person in red vest" --show-mask

# Multiple prompts on multiple images
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype q4f16 --batch 2 \
    --source assets/sam3-demo.jpg --source assets/bus.jpg \
    -p bus -p cat -p shoe -p cellphone -p person -p "short hair"

# Visual prompt (bbox)
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --dtype fp16 --source assets/sam3-demo.jpg \
    -p "visual;pos:480,290,110,360"

# Visual prompt: multi-boxes prompting(with positive and negative boxes)
cargo run -r -F sam3 -F cuda --example sam3 -- \
    --device cuda --source ./assets/sam3-demo.jpg \
    -p "visual;pos:480,290,110,360;neg:370,280,115,375"

# Text + negative box
cargo run -r -F sam3,cuda --example sam3 -- \
    --device cuda --source assets/000000136466.jpg \
    -p "handle;neg:40,183,278,21"

```

## Prompt Format

```
text;pos:x,y,w,h;neg:x,y,w,h
```

- `text` - Text description
- `pos:x,y,w,h` - Positive box (include region)
- `neg:x,y,w,h` - Negative box (exclude region)

## Results

![](https://github.com/jamjamjon/assets/releases/download/sam3/demo2.jpg)
![](https://github.com/jamjamjon/assets/releases/download/sam3/demo.jpg)
