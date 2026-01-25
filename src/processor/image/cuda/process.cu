// CUDA kernels for image preprocessing
// Supports: resize (bilinear), normalize, standardize, hwc2chw, letterbox

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

// =============================================================================
// Independent kernels (for flexibility and debugging)
// =============================================================================

/// RGB to BGR conversion (in-place capable)
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

/// HWC to CHW conversion with optional normalization
/// Input: [H, W, C] as u8 or f32
/// Output: [C, H, W] as f32
extern "C" __global__ void hwc2chw_u8(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    int channels,
    float scale,           // typically 1/255.0 for normalization, 1.0 otherwise
    const float* __restrict__ mean,  // can be nullptr
    const float* __restrict__ std    // can be nullptr
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int hw = height * width;
    int hwc_idx = (y * width + x) * channels;
    int chw_base = y * width + x;
    
    for (int c = 0; c < channels; ++c) {
        float val = static_cast<float>(input[hwc_idx + c]) * scale;
        
        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }
        
        output[c * hw + chw_base] = val;
    }
}

/// HWC to CHW conversion for f32 input
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
    
    for (int c = 0; c < channels; ++c) {
        float val = input[hwc_idx + c] * scale;
        
        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }
        
        output[c * hw + chw_base] = val;
    }
}

/// Bilinear resize kernel
/// Input: [H_in, W_in, C] as u8
/// Output: [H_out, W_out, C] as u8
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
    
    // Scale factors (align_corners=false style)
    float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
    
    // Source coordinates (center-aligned)
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Clamp to valid range
    x0 = max(0, min(x0, in_width - 1));
    x1 = max(0, min(x1, in_width - 1));
    y0 = max(0, min(y0, in_height - 1));
    y1 = max(0, min(y1, in_height - 1));
    
    float dx = src_x - floorf(src_x);
    float dy = src_y - floorf(src_y);
    
    // Bilinear interpolation for each channel
    for (int c = 0; c < channels; ++c) {
        float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
        
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;
        
        output[(y * out_width + x) * channels + c] = static_cast<unsigned char>(clamp(val + 0.5f, 0.0f, 255.0f));
    }
}

/// Bilinear resize with float output (for direct preprocessing)
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
    
    float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    x0 = max(0, min(x0, in_width - 1));
    x1 = max(0, min(x1, in_width - 1));
    y0 = max(0, min(y0, in_height - 1));
    y1 = max(0, min(y1, in_height - 1));
    
    float dx = src_x - floorf(src_x);
    float dy = src_y - floorf(src_y);
    
    for (int c = 0; c < channels; ++c) {
        float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
        
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;
        
        output[(y * out_width + x) * channels + c] = val;
    }
}

// =============================================================================
// Fused kernels (for maximum performance)
// =============================================================================

/// Fused: resize (bilinear) + normalize + standardize + hwc2chw
/// This is the main preprocessing kernel for most vision models
/// Input: [H_in, W_in, 3] as u8 (RGB image)
/// Output: [3, H_out, W_out] as f32 (NCHW tensor, without batch dim)
extern "C" __global__ void preprocess_resize_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    float scale,                      // 1/255.0 for normalize, 1.0 otherwise
    const float* __restrict__ mean,   // can be nullptr
    const float* __restrict__ std,    // can be nullptr
    int apply_unsigned
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= out_width || y >= out_height) return;
    
    const int channels = 3;
    
    // Bilinear interpolation coordinates
    float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
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
    
    // Process each channel
    #pragma unroll
    for (int c = 0; c < channels; ++c) {
        float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
        
        // Bilinear interpolation
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;

        // Match CPU path more closely: CPU resizes into u8 image first, then converts to f32.
        val = clamp(val + 0.5f, 0.0f, 255.0f);
        
        // Normalize
        val *= scale;
        
        // Standardize
        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }

        if (apply_unsigned) {
            val = fmaxf(val, 0.0f);
        }
        
        // Write to CHW layout
        output[c * hw + out_idx] = val;
    }
}

/// Fused: letterbox resize + normalize + standardize + hwc2chw
/// Maintains aspect ratio with padding
/// Input: [H_in, W_in, 3] as u8
/// Output: [3, H_out, W_out] as f32
extern "C" __global__ void preprocess_letterbox_nchw(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int resized_height,    // actual resized image height (before padding)
    int resized_width,     // actual resized image width (before padding)
    int pad_top,           // top padding
    int pad_left,          // left padding
    float padding_value,   // padding pixel value (normalized, e.g., 114/255.0)
    float scale,           // 1/255.0 for normalize
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
    
    // Check if this pixel is in the padding region
    int x_rel = x - pad_left;
    int y_rel = y - pad_top;
    
    bool is_padding = (x_rel < 0 || x_rel >= resized_width || 
                       y_rel < 0 || y_rel >= resized_height);
    
    if (is_padding) {
        // Write padding value
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
    } else {
        // Bilinear interpolation from source image
        float scale_x = static_cast<float>(in_width) / static_cast<float>(resized_width);
        float scale_y = static_cast<float>(in_height) / static_cast<float>(resized_height);
        
        float src_x = (x_rel + 0.5f) * scale_x - 0.5f;
        float src_y = (y_rel + 0.5f) * scale_y - 0.5f;
        
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
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
            float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
            float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
            float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
            float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
            
            float val = (1.0f - dx) * (1.0f - dy) * v00 +
                        dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 +
                        dx * dy * v11;

            val = clamp(val + 0.5f, 0.0f, 255.0f);
            
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
}

/// Fused: resize + normalize + standardize (keep NHWC layout)
/// For models that expect NHWC input
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
    
    float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
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
        float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
        
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;

        val = clamp(val + 0.5f, 0.0f, 255.0f);
        
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

/// Letterbox with NHWC output
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
    
    bool is_padding = (x_rel < 0 || x_rel >= resized_width || 
                       y_rel < 0 || y_rel >= resized_height);
    
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
    } else {
        float scale_x = static_cast<float>(in_width) / static_cast<float>(resized_width);
        float scale_y = static_cast<float>(in_height) / static_cast<float>(resized_height);
        
        float src_x = (x_rel + 0.5f) * scale_x - 0.5f;
        float src_y = (y_rel + 0.5f) * scale_y - 0.5f;
        
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
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
            float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
            float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
            float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
            float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
            
            float val = (1.0f - dx) * (1.0f - dy) * v00 +
                        dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 +
                        dx * dy * v11;

            // Match CPU path more closely (CPU resizes into u8 then converts)
            val = clamp(val + 0.5f, 0.0f, 255.0f);
            
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
}

/// FitAdaptive resize (no center padding, top-left aligned)
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
    
    // Top-left aligned (no centering)
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
    } else {
        float scale_x = static_cast<float>(in_width) / static_cast<float>(resized_width);
        float scale_y = static_cast<float>(in_height) / static_cast<float>(resized_height);
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
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
            float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
            float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
            float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
            float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
            
            float val = (1.0f - dx) * (1.0f - dy) * v00 +
                        dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 +
                        dx * dy * v11;

            val = clamp(val + 0.5f, 0.0f, 255.0f);
            
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
}

// =============================================================================
// Transform kernels for ImageTransform system
// =============================================================================

/// Pad image to multiple of window_size with constant fill
/// Input: [H, W, C] as u8
/// Output: [H_padded, W_padded, C] as u8
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

/// Pad image to multiple of window_size with reflect fill
/// Input: [H, W, C] as u8
/// Output: [H_padded, W_padded, C] as u8
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
    
    // Reflect padding (matches current CPU semantics: reflect including edge)
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

/// Pad image to multiple of window_size with replicate (edge) fill
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

/// Pad image to multiple of window_size with wrap fill
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

/// Pad image with fixed padding (top, bottom, left, right)
extern "C" __global__ void pad_fixed(
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

/// Center crop image
/// Input: [H, W, C] as u8
/// Output: [crop_H, crop_W, C] as u8
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
    
    // Match torchvision.transforms.functional.center_crop semantics.
    // - If crop is larger than input, treat missing pixels as 0 (implicit padding).
    // - If crop is smaller, crop offset uses round((in - crop) / 2.0) == (diff + 1) / 2 for integer diff.
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

/// Extract a single patch from image (for AnyRes)
/// Input: [H, W, C] as u8
/// Output: [patch_H, patch_W, C] as u8
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
    
    // Clamp to image bounds
    src_x = min(max(src_x, 0), in_width - 1);
    src_y = min(max(src_y, 0), in_height - 1);
    
    int in_idx = (src_y * in_width + src_x) * channels;
    int out_idx = (y * patch_width + x) * channels;
    
    for (int c = 0; c < channels; ++c) {
        output[out_idx + c] = input[in_idx + c];
    }
}

/// Extract multiple patches and stack into batch (for AnyRes SmolVLM/Moondream2)
/// Input: [H, W, C] as u8
/// Output: [N, patch_H, patch_W, C] as u8 where N = num_rows * num_cols (+ 1 for global if included)
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
    int include_global  // 1 = include resized global image as last patch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int patch_idx = blockIdx.z;
    
    int total_patches = num_rows * num_cols + (include_global ? 1 : 0);
    if (x >= patch_width || y >= patch_height || patch_idx >= total_patches) return;
    
    int out_idx = (patch_idx * patch_height * patch_width + y * patch_width + x) * channels;
    
    if (include_global && patch_idx == total_patches - 1) {
        // Global patch: bilinear resize from full image
        float scale_x = static_cast<float>(in_width) / static_cast<float>(patch_width);
        float scale_y = static_cast<float>(in_height) / static_cast<float>(patch_height);
        
        float src_x = (x + 0.5f) * scale_x - 0.5f;
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        x0 = max(0, min(x0, in_width - 1));
        x1 = max(0, min(x1, in_width - 1));
        y0 = max(0, min(y0, in_height - 1));
        y1 = max(0, min(y1, in_height - 1));
        
        float dx = src_x - floorf(src_x);
        float dy = src_y - floorf(src_y);
        
        for (int c = 0; c < channels; ++c) {
            float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
            float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
            float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
            float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
            
            float val = (1.0f - dx) * (1.0f - dy) * v00 +
                        dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 +
                        dx * dy * v11;
            
            output[out_idx + c] = static_cast<unsigned char>(clamp(val + 0.5f, 0.0f, 255.0f));
        }
    } else {
        // Regular patch: extract from grid
        int row = patch_idx / num_cols;
        int col = patch_idx % num_cols;
        
        int cell_height = in_height / num_rows;
        int cell_width = in_width / num_cols;
        
        int start_y = row * cell_height;
        int start_x = col * cell_width;
        
        // Bilinear resize from cell to patch size
        float scale_x = static_cast<float>(cell_width) / static_cast<float>(patch_width);
        float scale_y = static_cast<float>(cell_height) / static_cast<float>(patch_height);
        
        float src_x = (x + 0.5f) * scale_x - 0.5f + start_x;
        float src_y = (y + 0.5f) * scale_y - 0.5f + start_y;
        
        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        x0 = max(0, min(x0, in_width - 1));
        x1 = max(0, min(x1, in_width - 1));
        y0 = max(0, min(y0, in_height - 1));
        y1 = max(0, min(y1, in_height - 1));
        
        float dx = src_x - floorf(src_x);
        float dy = src_y - floorf(src_y);
        
        for (int c = 0; c < channels; ++c) {
            float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
            float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
            float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
            float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
            
            float val = (1.0f - dx) * (1.0f - dy) * v00 +
                        dx * (1.0f - dy) * v01 +
                        (1.0f - dx) * dy * v10 +
                        dx * dy * v11;
            
            output[out_idx + c] = static_cast<unsigned char>(clamp(val + 0.5f, 0.0f, 255.0f));
        }
    }
}

/// Batch extract patches with resize and final postprocess (NCHW output)
/// For AnyRes: extracts patches, resizes each to target size, applies normalize/standardize
/// Input: [H, W, C] as u8
/// Output: [N, C, target_H, target_W] as f32
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
    int out_base = patch_idx * patch_size + y * target_width + x;
    
    float src_x, src_y;
    
    if (include_global && patch_idx == total_patches - 1) {
        // Global patch
        float scale_x = static_cast<float>(in_width) / static_cast<float>(target_width);
        float scale_y = static_cast<float>(in_height) / static_cast<float>(target_height);
        src_x = (x + 0.5f) * scale_x - 0.5f;
        src_y = (y + 0.5f) * scale_y - 0.5f;
    } else {
        // Grid patch
        int row = patch_idx / num_cols;
        int col = patch_idx % num_cols;
        int cell_height = in_height / num_rows;
        int cell_width = in_width / num_cols;
        int start_y = row * cell_height;
        int start_x = col * cell_width;
        
        float scale_x = static_cast<float>(cell_width) / static_cast<float>(target_width);
        float scale_y = static_cast<float>(cell_height) / static_cast<float>(target_height);
        src_x = (x + 0.5f) * scale_x - 0.5f + start_x;
        src_y = (y + 0.5f) * scale_y - 0.5f + start_y;
    }
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
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
        float v00 = static_cast<float>(input[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(input[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(input[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(input[(y1 * in_width + x1) * channels + c]);
        
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;
        
        val = clamp(val + 0.5f, 0.0f, 255.0f);
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

// =============================================================================
// Batch processing kernels (optional, for when all images have same source size)
// =============================================================================

/// Batch preprocess: resize + normalize + hwc2chw for uniform-sized batch
/// Input: [N, H_in, W_in, 3] as u8
/// Output: [N, 3, H_out, W_out] as f32
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
    
    float scale_x = static_cast<float>(in_width) / static_cast<float>(out_width);
    float scale_y = static_cast<float>(in_height) / static_cast<float>(out_height);
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));
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
        float v00 = static_cast<float>(in_ptr[(y0 * in_width + x0) * channels + c]);
        float v01 = static_cast<float>(in_ptr[(y0 * in_width + x1) * channels + c]);
        float v10 = static_cast<float>(in_ptr[(y1 * in_width + x0) * channels + c]);
        float v11 = static_cast<float>(in_ptr[(y1 * in_width + x1) * channels + c]);
        
        float val = (1.0f - dx) * (1.0f - dy) * v00 +
                    dx * (1.0f - dy) * v01 +
                    (1.0f - dx) * dy * v10 +
                    dx * dy * v11;
        
        val *= scale;
        
        if (mean != nullptr && std != nullptr) {
            val = (val - mean[c]) / std[c];
        }
        
        out_ptr[c * hw + out_idx] = val;
    }
}
