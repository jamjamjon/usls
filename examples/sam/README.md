## Quick Start

```Shell

# SAM
cargo run -r -F cuda --example sam -- --device cuda --kind sam

# MobileSAM
cargo run -r -F cuda --example sam -- --device cuda --kind mobile-sam

# EdgeSAM
cargo run -r -F cuda --example sam -- --device cuda --kind edge-sam

# SAM-HQ
cargo run -r -F cuda --example sam -- --device cuda --kind sam-hq
```


## Results

![](https://github.com/jamjamjon/assets/releases/download/sam/demo-car.png)
![](https://github.com/jamjamjon/assets/releases/download/sam/demo-dog.png)
