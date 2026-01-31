# Installation

To use **usls** in your project, add it to your `Cargo.toml`.

=== "GitHub (Recommended)"

    ```toml
    [dependencies]
    usls = { git = "https://github.com/jamjamjon/usls", branch = "main" }
    ```

=== "Crates.io"

    ```toml
    [dependencies]
    usls = { version = "latest-version", features = [ "cuda" ] }
    ```


!!! tip "Available Cargo Features Selection"

    **usls** provides multiple Cargo features for different execution providers and capabilities.

    Select the features that match your hardware and use case.

    [:material-cog: View All Cargo Features →](../cargo-features/overview.md){ .md-button }


## Next Steps

<div class="grid cards" markdown>

-   :material-code-braces:{ .lg .middle } **Integration**

    ---

    Learn how to use usls in your code

    [Integrate →](integration.md)

</div>
