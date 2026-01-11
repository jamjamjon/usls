# Utils Module

Shared utilities for all examples.

## Contents

### `init_logging()`

Initializes the tracing subscriber for logging.

## Usage

```rust
#[path = "../utils/mod.rs"]
mod utils;

fn main() -> Result<()> {
    utils::init_logging();
}
```
