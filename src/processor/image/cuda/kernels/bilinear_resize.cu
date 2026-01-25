extern "C" __global__ void resize_bilinear_u8(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float scale_x = (float)in_width / (float)out_width;
    float scale_y = (float)in_height / (float)out_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, in_width - 1));
    x1 = max(0, min(x1, in_width - 1));
    y0 = max(0, min(y0, in_height - 1));
    y1 = max(0, min(y1, in_height - 1));

    float dx = src_x - floorf(src_x);
    float dy = src_y - floorf(src_y);

    int out_base = (y * out_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        output[out_base + c] = (unsigned char)val;
    }
}

extern "C" __global__ void resize_bilinear_u8_to_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float scale_x = (float)in_width / (float)out_width;
    float scale_y = (float)in_height / (float)out_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, in_width - 1));
    x1 = max(0, min(x1, in_width - 1));
    y0 = max(0, min(y0, in_height - 1));
    y1 = max(0, min(y1, in_height - 1));

    float dx = src_x - floorf(src_x);
    float dy = src_y - floorf(src_y);

    int out_base = (y * out_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        output[out_base + c] = val;
    }
}
