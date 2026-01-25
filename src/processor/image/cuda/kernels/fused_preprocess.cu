extern "C" __global__ void preprocess_resize_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int channels = 3;

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

    int hw = out_height * out_width;
    int out_idx = y * out_width + x;

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }

        output[c * hw + out_idx] = val;
    }
}

extern "C" __global__ void preprocess_letterbox_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int resized_height,
    int resized_width,
    int pad_top,
    int pad_left,
    float padding_value,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int channels = 3;
    int hw = out_height * out_width;
    int out_idx = y * out_width + x;

    int x_rel = x - pad_left;
    int y_rel = y - pad_top;

    bool is_padding = (x_rel < 0 || x_rel >= resized_width || y_rel < 0 || y_rel >= resized_height);

    if (is_padding) {
        #pragma unroll
        for (int c = 0; c < channels; ++c) {
            float val = padding_value;
            if (mean != nullptr && std != nullptr) {
                val = (val - mean[c]) / std[c];
            }
            if (apply_unsigned) {
                val = fmaxf(val, 0.0f);
            }
            output[c * hw + out_idx] = val;
        }
        return;
    }

    float scale_x = (float)in_width / (float)resized_width;
    float scale_y = (float)in_height / (float)resized_height;

    float src_x = (x_rel + 0.5f) * scale_x - 0.5f;
    float src_y = (y_rel + 0.5f) * scale_y - 0.5f;

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

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }

        output[c * hw + out_idx] = val;
    }
}

extern "C" __global__ void preprocess_resize_nhwc(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int channels = 3;

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

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }

        output[out_base + c] = val;
    }
}

extern "C" __global__ void preprocess_letterbox_nhwc(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int resized_height,
    int resized_width,
    int pad_top,
    int pad_left,
    float padding_value,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int channels = 3;
    int out_base = (y * out_width + x) * channels;

    int x_rel = x - pad_left;
    int y_rel = y - pad_top;

    bool is_padding = (x_rel < 0 || x_rel >= resized_width || y_rel < 0 || y_rel >= resized_height);

    if (is_padding) {
        #pragma unroll
        for (int c = 0; c < channels; ++c) {
            float val = padding_value;
            if (mean != nullptr && std != nullptr) {
                val = (val - mean[c]) / std[c];
            }
            if (apply_unsigned) {
                val = fmaxf(val, 0.0f);
            }
            output[out_base + c] = val;
        }
        return;
    }

    float scale_x = (float)in_width / (float)resized_width;
    float scale_y = (float)in_height / (float)resized_height;

    float src_x = (x_rel + 0.5f) * scale_x - 0.5f;
    float src_y = (y_rel + 0.5f) * scale_y - 0.5f;

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

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }

        output[out_base + c] = val;
    }
}

extern "C" __global__ void preprocess_fit_adaptive_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int resized_height,
    int resized_width,
    float padding_value,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    const int channels = 3;
    int hw = out_height * out_width;
    int out_idx = y * out_width + x;

    bool is_padding = (x >= resized_width || y >= resized_height);

    if (is_padding) {
        #pragma unroll
        for (int c = 0; c < channels; ++c) {
            float val = padding_value;
            if (mean != nullptr && std != nullptr) {
                val = (val - mean[c]) / std[c];
            }
            if (apply_unsigned) {
                val = fmaxf(val, 0.0f);
            }
            output[c * hw + out_idx] = val;
        }
        return;
    }

    float scale_x = (float)in_width / (float)resized_width;
    float scale_y = (float)in_height / (float)resized_height;

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

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)input[(y0 * in_width + x0) * channels + c];
        float v01 = (float)input[(y0 * in_width + x1) * channels + c];
        float v10 = (float)input[(y1 * in_width + x0) * channels + c];
        float v11 = (float)input[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }

        output[c * hw + out_idx] = val;
    }
}

extern "C" __global__ void preprocess_batch_resize_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= out_width || y >= out_height || n >= batch_size) return;

    const int channels = 3;
    int in_image_size = in_height * in_width * channels;
    int out_image_size = out_height * out_width * channels;

    const unsigned char* in_ptr = input + n * in_image_size;
    float* out_ptr = output + n * out_image_size;

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

    int hw = out_height * out_width;
    int out_idx = y * out_width + x;

    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = (float)in_ptr[(y0 * in_width + x0) * channels + c];
        float v01 = (float)in_ptr[(y0 * in_width + x1) * channels + c];
        float v10 = (float)in_ptr[(y1 * in_width + x0) * channels + c];
        float v11 = (float)in_ptr[(y1 * in_width + x1) * channels + c];

        float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 + dx * dy * v11;

        val *= scale;

        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        out_ptr[c * hw + out_idx] = val;
    }
}
