// ST-DeInt zero-copy経路用のRGB weaveカーネル。

__kernel void onnx_deint_weave_rgb(
    __global const float *input,
    __global const float *restorations,
    __global float *output,
    const int width,
    const int height,
    const int frameA) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    const int plane = width * height;
    const int halfPlane = plane / 2;
    const int index = y * width + x;
    const int useInput = ((y & 1) == (frameA ? 0 : 1));
    const int restoreFrameOffset = frameA ? 0 : 3 * halfPlane;
    const int restoreIndex = restoreFrameOffset + (y / 2) * width + x;
    for (int channel = 0; channel < 3; channel++) {
        output[channel * plane + index] = useInput
            ? input[channel * plane + index]
            : restorations[channel * halfPlane + restoreIndex];
    }
}

// CUDA版と同じ行写像でフィールドを上下に配置し、推論結果を元の行順へ戻す。
__kernel void onnx_deint_reorder_fields(
    __global const float *input,
    __global float *output,
    const int width,
    const int height,
    const int stack) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int halfHeight = height / 2;
    const int sourceY = stack ? (y % halfHeight) * 2 + y / halfHeight : y / 2 + (y & 1) * halfHeight;
    const int plane = width * height;
    for (int channel = 0; channel < 3; channel++) {
        const float value = input[channel * plane + sourceY * width + x];
        output[channel * plane + y * width + x] = stack ? value : clamp(value, 0.0f, 1.0f);
    }
}
