__device__ __forceinline__ float clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

// fast_image_resize uses a precomputed CLIP8_LOOKUPS table that effectively maps
// (v >> precision) into [0, 255] for any value within a safe range.
// This saturating clip is equivalent for all representable u8 outputs.
__device__ __forceinline__ unsigned char clip_u8(int v_shifted) {
    if (v_shifted <= 0) return (unsigned char)0;
    if (v_shifted >= 255) return (unsigned char)255;
    return (unsigned char)v_shifted;
}

extern "C" __global__ void rgb2bgr(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int height,
    int width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    output[idx] = b;
    output[idx + 1] = g;
    output[idx + 2] = r;
}

extern "C" __global__ void hwc2chw_u8(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    int channels,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    int hwc_idx = (y * width + x) * channels;
    int chw_base = y * width + x;

    if (channels == 3) {
        float v0 = (float)input[hwc_idx + 0] * scale;
        float v1 = (float)input[hwc_idx + 1] * scale;
        float v2 = (float)input[hwc_idx + 2] * scale;

        if (mean != nullptr && std != nullptr) {
            v0 = (v0 - mean[0]) / std[0];
            v1 = (v1 - mean[1]) / std[1];
            v2 = (v2 - mean[2]) / std[2];
        }

        output[0 * hw + chw_base] = v0;
        output[1 * hw + chw_base] = v1;
        output[2 * hw + chw_base] = v2;
        return;
    }

    for (int c = 0; c < channels; ++c) {
        float val = (float)input[hwc_idx + c] * scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        output[c * hw + chw_base] = val;
    }
}

extern "C" __global__ void hwc2chw_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    int channels,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    int hwc_idx = (y * width + x) * channels;
    int chw_base = y * width + x;

    if (channels == 3) {
        float v0 = input[hwc_idx + 0] * scale;
        float v1 = input[hwc_idx + 1] * scale;
        float v2 = input[hwc_idx + 2] * scale;

        if (mean != nullptr && std != nullptr) {
            v0 = (v0 - mean[0]) / std[0];
            v1 = (v1 - mean[1]) / std[1];
            v2 = (v2 - mean[2]) / std[2];
        }

        output[0 * hw + chw_base] = v0;
        output[1 * hw + chw_base] = v1;
        output[2 * hw + chw_base] = v2;
        return;
    }

    for (int c = 0; c < channels; ++c) {
        float val = input[hwc_idx + c] * scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        output[c * hw + chw_base] = val;
    }
}
