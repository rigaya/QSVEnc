#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef clamp
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#endif

__kernel void kernel_dummy_load(
    __global uchar *restrict buf, const int N, const int M, const float a, const float b) {
    const int ix = get_global_id(0);
    if (ix < N) {
        float value = (float)(buf[ix] * (1.0f/255.0f));
        for (int i = 0; i < M; i++) {
            value += a;
            value -= b;
        }
        buf[ix] = clamp((uchar)(value * 255.0f), 0, 255);
    }
}
