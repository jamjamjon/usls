# Hub

`Hub` downloads and caches files from **GitHub Releases** and **Hugging Face** repositories.

## Supported Formats

| Source | Format | Example |
|--------|--------|---------|
| Local file | File path | `"./model.onnx"` |
| GitHub Release | `<tag>/<file>` | `"yolo/v5-n-det.onnx"` |
| GitHub Release URL | Full URL | `"https://github.com/<owner>/<repo>/releases/download/<tag>/<file>"` |
| HF (inline) | `<owner>/<repo>/<file>` | `"BAAI/bge-m3/tokenizer.json"` |
| HF (dedicated) | `<file>` via `from_hf` | `"onnx/model.onnx"` |
| HF URL | Full URL (`resolve`/`blob`) | `"https://huggingface.co/<owner>/<repo>/blob/main/<file>"` |

!!! tip "HF Endpoint"
    By default, Hugging Face downloads use `https://huggingface.co`.  

    Set the `HF_ENDPOINT` environment variable to use a mirror:
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```

## GitHub Release

!!! example "Default Repository"
    Download files from the default GitHub repository (`jamjamjon/assets`):

    ```rust
    let path = Hub::default().try_fetch("images/bus.jpg")?;
    ```

!!! example "Custom Repository"
    ```rust
    let mut hub = Hub::new("owner", "repo");
    let path = hub.try_fetch("<tag>/<file>")?;
    ```

!!! example "Direct GitHub URL"
    ```rust
    let path = Hub::default().try_fetch(
        "https://github.com/<owner>/<repo>/releases/download/<tag>/<file>"
    )?;
    ```

## Hugging Face

!!! example "Inline Path (Recommended)"
    Use `<owner>/<repo>/<file>` format directly — no extra setup needed:

    ```rust
    let path = Hub::default().try_fetch("<owner>/<repo>/<folder>/<file>")?;
    ```

!!! example "Dedicated Hub"
    Bind a Hub to a specific HF repository:

    ```rust
    let mut hub = Hub::from_hf("<owner>", "<repo>")?;
    let path = hub.try_fetch("<file>")?;
    let path = hub.try_fetch("<folder>/<file>")?;
    ```

!!! example "Direct HF URL"
    Supports both `/resolve/` and `/blob/` URLs:

    ```rust
    let path = Hub::default().try_fetch(
        "https://huggingface.co/<owner>/<repo>/blob/main/<file>"
    )?;
    ```

## Repository Info

!!! example "Inspect Repository"
    ```rust
    Hub::default().info()?;                         // GitHub releases
    Hub::from_hf("<owner>", "<repo>")?.info()?;     // HF file tree with sizes
    ```

## Caching

!!! info "Cache Behavior"
    - Files are cached locally after the first download (`~/.cache/usls/` or similar).
    - GitHub release metadata: TTL-based (default 10 min, configurable via `with_ttl`).
    - Failed or incomplete downloads are discarded (atomic write via temp files).
