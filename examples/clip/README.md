This demo showcases how to use [CLIP](https://github.com/openai/CLIP) to compute similarity between texts and images, which can be employed for image-to-text or text-to-image retrieval tasks.

## Quick Start

```shell
cargo run -r --example clip
```

## Results

```shell
(90.11472%) ./examples/clip/images/carrot.jpg => 几个胡萝卜 
[0.04573484, 0.0048218793, 0.0011618224, 0.90114725, 0.0036694852, 0.031348046, 0.0121166315]

(94.07785%) ./examples/clip/images/peoples.jpg => Some people holding wine glasses in a restaurant 
[0.050406333, 0.0011632168, 0.0019338318, 0.0013227565, 0.003916758, 0.00047858112, 0.9407785]

(86.59852%) ./examples/clip/images/doll.jpg => There is a doll with red hair and a clock on a table 
[0.07032883, 0.00053773675, 0.0006372929, 0.06066096, 0.0007378078, 0.8659852, 0.0011121632]
```