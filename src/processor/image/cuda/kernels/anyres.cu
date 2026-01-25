extern "C" __global__ void extract_patch(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int patch_height,
    int patch_width,
    int start_y,
    int start_x,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= patch_width || y >= patch_height) return;

    int src_x = start_x + x;
    int src_y = start_y + y;

    src_x = min(max(src_x, 0), in_width - 1);
    src_y = min(max(src_y, 0), in_height - 1);

    int in_idx = (src_y * in_width + src_x) * channels;
    int out_idx = (y * patch_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void extract_patches_grid(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int patch_height,
    int patch_width,
    int num_rows,
    int num_cols,
    int channels,
    int include_global
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int patch_idx = blockIdx.z;

    int total_patches = num_rows * num_cols + (include_global ? 1 : 0);
    if (x >= patch_width || y >= patch_height || patch_idx >= total_patches) return;

    int out_idx = (patch_idx * patch_height * patch_width + y * patch_width + x) * channels;

    if (include_global && patch_idx == total_patches - 1) {
        float scale_x = (float)in_width / (float)patch_width;
        float scale_y = (float)in_height / (float)patch_height;

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

        for (int c = 0; c < channels; ++c) {
            float v00 = (float)input[(y0 * in_width + x0) * channels + c];
            float v01 = (float)input[(y0 * in_width + x1) * channels + c];
            float v10 = (float)input[(y1 * in_width + x0) * channels + c];
            float v11 = (float)input[(y1 * in_width + x1) * channels + c];

            float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 + dx * dy * v11;

            val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
            output[out_idx + c] = (unsigned char)val;
        }
    } else {
        int row = patch_idx / num_cols;
        int col = patch_idx % num_cols;

        int cell_height = in_height / num_rows;
        int cell_width = in_width / num_cols;

        int start_y = row * cell_height;
        int start_x = col * cell_width;

        float scale_x = (float)cell_width / (float)patch_width;
        float scale_y = (float)cell_height / (float)patch_height;

        float src_x = (x + 0.5f) * scale_x - 0.5f + start_x;
        float src_y = (y + 0.5f) * scale_y - 0.5f + start_y;

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

        for (int c = 0; c < channels; ++c) {
            float v00 = (float)input[(y0 * in_width + x0) * channels + c];
            float v01 = (float)input[(y0 * in_width + x1) * channels + c];
            float v10 = (float)input[(y1 * in_width + x0) * channels + c];
            float v11 = (float)input[(y1 * in_width + x1) * channels + c];

            float val = (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 + dx * dy * v11;

            val = fminf(fmaxf(val + 0.5f, 0.0f), 255.0f);
            output[out_idx + c] = (unsigned char)val;
        }
    }
}

extern "C" __global__ void dynres_patches_to_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int target_height,
    int target_width,
    int num_rows,
    int num_cols,
    int include_global,
    float scale,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int patch_idx = blockIdx.z;

    int total_patches = num_rows * num_cols + (include_global ? 1 : 0);
    if (x >= target_width || y >= target_height || patch_idx >= total_patches) return;

    const int channels = 3;
    int hw = target_height * target_width;
    int patch_size = hw * channels;

    float src_x;
    float src_y;

    if (include_global && patch_idx == total_patches - 1) {
        float scale_x = (float)in_width / (float)target_width;
        float scale_y = (float)in_height / (float)target_height;
        src_x = (x + 0.5f) * scale_x - 0.5f;
        src_y = (y + 0.5f) * scale_y - 0.5f;
    } else {
        int row = patch_idx / num_cols;
        int col = patch_idx % num_cols;
        int cell_height = in_height / num_rows;
        int cell_width = in_width / num_cols;
        int start_y = row * cell_height;
        int start_x = col * cell_width;

        float scale_x = (float)cell_width / (float)target_width;
        float scale_y = (float)cell_height / (float)target_height;
        src_x = (x + 0.5f) * scale_x - 0.5f + start_x;
        src_y = (y + 0.5f) * scale_y - 0.5f + start_y;
    }

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

        output[patch_idx * patch_size + c * hw + y * target_width + x] = val;
    }
}
