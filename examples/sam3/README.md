
### Quick Start

```bash
# Text prompt
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype q4f16 --source ./assets/sam3-demo.jpg -p shoe
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype bnb4 --source ./assets/sam3-demo.jpg -p "person in red vest"
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype q8 --source ./assets/sam3-demo.jpg -p "boy in blue vest"

# Visual prompt: a single bbox
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --source ./assets/sam3-demo.jpg -p "visual;pos:480,290,110,360"

# Visual prompt: multi-boxes prompting(with positive and negative boxes)
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda  --source ./assets/sam3-demo.jpg -p "visual;pos:480,290,110,360;neg:370,280,115,375"

# Text + negative box
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype fp16 --source ./assets/000000136466.jpg -p "handle"
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype fp16 --source ./assets/000000136466.jpg -p "handle;neg:40,183,278,21"

# Multiple prompts (Queries)
cargo run -r -F sam3 -F cuda --example sam3 -- --device cuda --dtype fp16 --source ./assets/sam3-demo.jpg --source ./assets/bus.jpg -p shoe -p face -p person
```



### Prompt Format

```
"text;pos:x,y,w,h;neg:x,y,w,h"
```

- `text`: Text description
- `pos:x,y,w,h`: Positive box (find similar)
- `neg:x,y,w,h`: Negative box (exclude region)


### Results

![](https://github.com/jamjamjon/assets/releases/download/sam3/demo.jpg)
![](https://github.com/jamjamjon/assets/releases/download/sam3/demo2.jpg)
