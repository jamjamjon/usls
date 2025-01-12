This demo shows how to use [BLIP](https://arxiv.org/abs/2201.12086) to do conditional or unconditional image captioning.

## Quick Start

```shell
cargo run -r -F cuda --example blip -- --device cuda:0 --source images/dog.jpg --source ./assets/bus.jpg --source images/green-car.jpg
```

## Results

```shell
Unconditional: Ys([Y { Texts: [Text("a dog running through a field of grass")] }, Y { Texts: [Text("a group of people walking around a bus")] }, Y { Texts: [Text("a green volkswagen beetle parked in front of a yellow building")] }])
Conditional: Ys([Y { Texts: [Text("this image depicting a dog running in a field")] }, Y { Texts: [Text("this image depict a bus in barcelona")] }, Y { Texts: [Text("this image depict a blue volkswagen beetle parked in a street in havana, cuba")] }])
```
