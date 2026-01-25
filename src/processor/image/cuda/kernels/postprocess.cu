extern "C" __global__ void postprocess_pad_u8_to_nchw_f32(
    const unsigned char* __restrict__ resized,
    float* __restrict__ output,
    int resized_height,
    int resized_width,
    int out_height,
    int out_width,
    int pad_top,
    int pad_left,
    unsigned char padding_value,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int x_rel = x - pad_left;
    int y_rel = y - pad_top;
    bool is_padding = (x_rel < 0 || x_rel >= resized_width || y_rel < 0 || y_rel >= resized_height);

    int hw = out_height * out_width;
    int out_idx = y * out_width + x;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v;
        if (is_padding) {
            v = (float)padding_value;
        } else {
            int in_idx = (y_rel * resized_width + x_rel) * 3 + c;
            v = (float)resized[in_idx];
        }

        v *= scale;
        if (mean != nullptr && std != nullptr) {
            v = (v - mean[c]) / std[c];
        }
        if (apply_unsigned) {
            v = fmaxf(v, 0.0f);
        }
        output[c * hw + out_idx] = v;
    }
}

extern "C" __global__ void postprocess_pad_u8_to_nhwc_f32(
    const unsigned char* __restrict__ resized,
    float* __restrict__ output,
    int resized_height,
    int resized_width,
    int out_height,
    int out_width,
    int pad_top,
    int pad_left,
    unsigned char padding_value,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int x_rel = x - pad_left;
    int y_rel = y - pad_top;
    bool is_padding = (x_rel < 0 || x_rel >= resized_width || y_rel < 0 || y_rel >= resized_height);

    int out_base = (y * out_width + x) * 3;

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v;
        if (is_padding) {
            v = (float)padding_value;
        } else {
            int in_idx = (y_rel * resized_width + x_rel) * 3 + c;
            v = (float)resized[in_idx];
        }

        v *= scale;
        if (mean != nullptr && std != nullptr) {
            v = (v - mean[c]) / std[c];
        }
        if (apply_unsigned) {
            v = fmaxf(v, 0.0f);
        }
        output[out_base + c] = v;
    }
}
