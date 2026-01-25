## Example

**video + directory(image + video)**

```bash
cargo run -r -F video,viewer --example imshow -- --source "../7.mp4 | ./assets" --save 
```
> save frames to video
> Note: For multiple videos, frames will be saved to separate files. And image will not be saved to video.

**Num of frames to skip**
```bash
cargo run -r -F video,viewer --example imshow -- --source "../7.mp4 | ./assets" --save --nfv-skip 5  
```
