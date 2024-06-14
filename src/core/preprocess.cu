extern "C" __global__ void rgb2bgr(int* xs, int* ys, const int h, const int w) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = (x + y * w) * 3;
    if (x < w && y < h) {
        ys[tid] = xs[tid+ 2];
        ys[tid + 1] = xs[tid + 1];
        ys[tid + 2] = xs[tid];
    }
}

extern "C" __global__ void normalize(float* xs, float* ys, int h, int w, float* means, float* stds) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = (x + y * w) * 3;
    if (x < w && y < h) {
        ys[tid] = (xs[tid] - means[0]) / stds[0];
        ys[tid + 1] = (xs[tid + 1] - means[1]) / stds[1];
        ys[tid + 2] = (xs[tid + 2] - means[2]) / stds[2];
    }
}


extern "C" __global__ void hwc2chw(int* xs, int* ys, int h, int w) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * w;
    if (x < w && y < h) {
        ys[tid] = xs[tid * 3];
        ys[tid + h * w] = xs[tid * 3 + 1];
        ys[tid + h * w * 2] = xs[tid * 3 + 2];
    }
}


extern "C" __global__ void resize_bilinear(const float* input, float* output, int in_width, int in_height, int out_width, int out_height, int num_channel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;
    
    // align_corners
    float scale_x = static_cast<float>(in_width - 1) / static_cast<float>(out_width - 1);
    float scale_y = static_cast<float>(in_height - 1) / static_cast<float>(out_height - 1);
    float src_x = x * scale_x;
    float src_y = y * scale_y;
    int x0 = src_x;
    int y0 = src_y;
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);
    float dx = src_x - x0;
    float dy = src_y - y0;
    for (int c = 0; c < num_channel; ++c) {
        float value = (1 - dx) * (1 - dy) * input[(y0 * in_width + x0) * num_channel + c] +
                      dx * (1 - dy) * input[(y0 * in_width + x1) * num_channel + c] +
                      (1 - dx) * dy * input[(y1 * in_width + x0) * num_channel + c] +
                      dx * dy * input[(y1 * in_width + x1) * num_channel + c];
        output[(y * out_width + x) * num_channel + c] = static_cast<float>(value);
    }
}
