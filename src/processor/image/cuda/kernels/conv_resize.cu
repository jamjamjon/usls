extern "C" __global__ void conv_vert_u8x3(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int offset_x,
    const int* __restrict__ starts,
    const int* __restrict__ sizes,
    const int* __restrict__ offsets,
    const short* __restrict__ coeffs,
    int precision
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    int start = starts[y];
    int size = sizes[y];
    int off = offsets[y];

    int initial = 1 << (precision - 1);
    int ss0 = initial;
    int ss1 = initial;
    int ss2 = initial;

    int src_x = x + offset_x;
    const short* kptr = coeffs + off;

    for (int i = 0; i < size; ++i) {
        int src_y = start + i;
        int in_idx = (src_y * in_width + src_x) * 3;
        int k = (int)kptr[i];
        ss0 += (int)input[in_idx + 0] * k;
        ss1 += (int)input[in_idx + 1] * k;
        ss2 += (int)input[in_idx + 2] * k;
    }

    int out_idx = (y * out_width + x) * 3;
    output[out_idx + 0] = clip_u8(ss0 >> precision);
    output[out_idx + 1] = clip_u8(ss1 >> precision);
    output[out_idx + 2] = clip_u8(ss2 >> precision);
}

extern "C" __global__ void resize_nearest_u8x3(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= out_height) return;

    float scale_x = (float)in_width / (float)out_width;
    float scale_y = (float)in_height / (float)out_height;

    int src_x = (int)((x + 0.5f) * scale_x);
    int src_y = (int)((y + 0.5f) * scale_y);

    if (src_x < 0) src_x = 0;
    if (src_x > in_width - 1) src_x = in_width - 1;
    if (src_y < 0) src_y = 0;
    if (src_y > in_height - 1) src_y = in_height - 1;

    int in_idx = (src_y * in_width + src_x) * 3;
    int out_idx = (y * out_width + x) * 3;
    output[out_idx + 0] = input[in_idx + 0];
    output[out_idx + 1] = input[in_idx + 1];
    output[out_idx + 2] = input[in_idx + 2];
}

extern "C" __global__ void conv_horiz_u8x3(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int height,
    int in_width,
    int out_width,
    const int* __restrict__ starts,
    const int* __restrict__ sizes,
    const int* __restrict__ offsets,
    const short* __restrict__ coeffs,
    int precision
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= height) return;

    int start = starts[x];
    int size = sizes[x];
    int off = offsets[x];

    int initial = 1 << (precision - 1);
    int ss0 = initial;
    int ss1 = initial;
    int ss2 = initial;

    const short* kptr = coeffs + off;
    int base = y * in_width;
    for (int i = 0; i < size; ++i) {
        int src_x = start + i;
        int in_idx = (base + src_x) * 3;
        int k = (int)kptr[i];
        ss0 += (int)input[in_idx + 0] * k;
        ss1 += (int)input[in_idx + 1] * k;
        ss2 += (int)input[in_idx + 2] * k;
    }

    int out_idx = (y * out_width + x) * 3;
    output[out_idx + 0] = clip_u8(ss0 >> precision);
    output[out_idx + 1] = clip_u8(ss1 >> precision);
    output[out_idx + 2] = clip_u8(ss2 >> precision);
}
