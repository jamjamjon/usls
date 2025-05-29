This demo showcases how to use [CLIP](https://github.com/openai/CLIP) to compute similarity between texts and images, which can be employed for image-to-text or text-to-image retrieval tasks.

## Quick Start

```shell
cargo run -r -F cuda --example clip -- --device cuda:0
```

## Results

```shell
[99.999428%] (examples/clip/images/carrot.jpg) <=> (A picture of some carrots.)
[100.000000%] (examples/clip/images/doll.jpg) <=> (There is a doll with red hair and a clock on a table.)
[99.990738%] (examples/clip/images/drink.jpg) <=> (Some people holding wine glasses in a restaurant.)
```
