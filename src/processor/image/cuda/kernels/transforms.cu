extern "C" __global__ void pad_to_multiple_constant(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels,
    unsigned char fill_value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int out_idx = (y * out_width + x) * channels;

    if (x < in_width && y < in_height) {
        int in_idx = (y * in_width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            output[out_idx + c] = input[in_idx + c];
        }
    } else {
        for (int c = 0; c < channels; ++c) {
            output[out_idx + c] = fill_value;
        }
    }
}

extern "C" __global__ void pad_to_multiple_reflect(
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

    int src_x = x;
    int src_y = y;

    if (src_x >= in_width) {
        int dx = src_x - in_width;
        src_x = in_width - 1 - (dx % in_width);
    }
    if (src_y >= in_height) {
        int dy = src_y - in_height;
        src_y = in_height - 1 - (dy % in_height);
    }

    int in_idx = (src_y * in_width + src_x) * channels;
    int out_idx = (y * out_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

__device__ __forceinline__ int mod_pos_i32(int a, int b) {
    int m = a % b;
    return m < 0 ? m + b : m;
}

__device__ __forceinline__ int reflect_index_i32(int i, int n) {
    if (n <= 1) return 0;
    if (i < 0) {
        return ((-i - 1) % n);
    }
    if (i >= n) {
        int dx = i - n;
        return n - 1 - (dx % n);
    }
    return i;
}

extern "C" __global__ void pad_fixed_constant(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels,
    int pad_top,
    int pad_left,
    unsigned char fill_value
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int out_idx = (y * out_width + x) * channels;
    int src_x = x - pad_left;
    int src_y = y - pad_top;

    if (src_x >= 0 && src_x < in_width && src_y >= 0 && src_y < in_height) {
        int in_idx = (src_y * in_width + src_x) * channels;
        for (int c = 0; c < channels; ++c) {
            output[out_idx + c] = input[in_idx + c];
        }
    } else {
        for (int c = 0; c < channels; ++c) {
            output[out_idx + c] = fill_value;
        }
    }
}

extern "C" __global__ void pad_fixed_reflect(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels,
    int pad_top,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int out_idx = (y * out_width + x) * channels;
    int src_x = reflect_index_i32(x - pad_left, in_width);
    int src_y = reflect_index_i32(y - pad_top, in_height);
    int in_idx = (src_y * in_width + src_x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void pad_fixed_replicate(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels,
    int pad_top,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int out_idx = (y * out_width + x) * channels;
    int src_x = x - pad_left;
    int src_y = y - pad_top;
    src_x = max(0, min(src_x, in_width - 1));
    src_y = max(0, min(src_y, in_height - 1));
    int in_idx = (src_y * in_width + src_x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void pad_fixed_wrap(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int channels,
    int pad_top,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int out_idx = (y * out_width + x) * channels;
    int src_x = mod_pos_i32(x - pad_left, in_width);
    int src_y = mod_pos_i32(y - pad_top, in_height);
    int in_idx = (src_y * in_width + src_x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void pad_to_multiple_replicate(
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

    int src_x = min(x, in_width - 1);
    int src_y = min(y, in_height - 1);

    int in_idx = (src_y * in_width + src_x) * channels;
    int out_idx = (y * out_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void pad_to_multiple_wrap(
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

    int src_x = x % in_width;
    int src_y = y % in_height;

    int in_idx = (src_y * in_width + src_x) * channels;
    int out_idx = (y * out_width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

extern "C" __global__ void crop_center(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int crop_height,
    int crop_width,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= crop_width || y >= crop_height) return;

    int offset_x;
    int offset_y;

    if (crop_width > in_width) {
        offset_x = -((crop_width - in_width) / 2);
    } else {
        offset_x = (in_width - crop_width + 1) / 2;
    }

    if (crop_height > in_height) {
        offset_y = -((crop_height - in_height) / 2);
    } else {
        offset_y = (in_height - crop_height + 1) / 2;
    }

    int src_x = x + offset_x;
    int src_y = y + offset_y;

    int out_idx = (y * crop_width + x) * channels;

    if (src_x < 0 || src_x >= in_width || src_y < 0 || src_y >= in_height) {
        for (int c = 0; c < channels; ++c) {
            output[out_idx + c] = 0;
        }
        return;
    }

    int in_idx = (src_y * in_width + src_x) * channels;
    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}
