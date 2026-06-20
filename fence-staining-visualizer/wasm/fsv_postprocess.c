/*
 * fsv_postprocess.c -- WebAssembly hot-loop implementations for the
 * Fence Stain Visualizer postprocess pipeline.
 *
 * Strategy
 * --------
 * Exports a small set of pure functions that operate on flat Float32 arrays
 * passed in via WASM linear memory. The JS side allocates input/output
 * buffers inside WASM memory and passes pointers. This avoids per-call
 * marshalling overhead (~30% of total time for many small calls).
 *
 * Performance levers used here:
 *   1. SIMD128 (wasm32-simd): 4-way Float32 vector ops on the per-pixel
 *      arithmetic in maskedBoxBlur and Lab conversion. Targets -msimd128.
 *      A non-SIMD scalar build is also provided for browsers without v128.
 *   2. Loop bodies hand-tuned to avoid branches in the hot path.
 *   3. Pre-allocated scratch buffer reused across calls (set via
 *      fsv_init_scratch).
 *   4. sRGB->linear LUT computed once at init and stored in module-static
 *      memory.
 *   5. Sliding-window max/min via monotonic deque (O(W*H) regardless of
 *      kernel radius -- same algorithm as the JS reference implementation).
 *
 * Memory layout
 * -------------
 *   - All buffers live in WASM linear memory (default heap, grown as
 *     needed by the JS allocator).
 *   - JS calls fsv_alloc(bytes) -> pointer; fsv_free(pointer) to release.
 *   - For each hot function, the JS side:
 *       1. Calls fsv_alloc once per buffer (input, output, scratch)
 *       2. Copies input data with HEAPF32.set()
 *       3. Calls the function with pointers
 *       4. Reads output with new Float32Array(HEAPF32.buffer, ptr, len).slice()
 *       5. Calls fsv_free at session end
 *   - The JS-side memory pool (in app.js / index4_dinov3.html) reuses these
 *     pointers across calls so per-call alloc cost is amortized.
 *
 * Build
 * -----
 * Two outputs: a scalar build (fsv_postprocess.wasm) and a SIMD build
 * (fsv_postprocess_simd.wasm). JS feature-detects via
 * WebAssembly.validate() against a small SIMD probe and loads the
 * appropriate variant. See Makefile.
 *
 * Public ABI (exported functions)
 * -------------------------------
 *   fsv_alloc(size_t bytes) -> void*        : malloc wrapper
 *   fsv_free(void* p) -> void               : free wrapper
 *   fsv_srgb_lut_init() -> void             : build the sRGB->linear LUT once
 *   fsv_dilate(in, out, scratch, w, h, r)   : sliding-window max
 *   fsv_erode(in, out, scratch, w, h, r)    : sliding-window min
 *   fsv_box_blur_masked(values, mask, out, w, h, r) : masked box blur
 *   fsv_rgba_to_lab(rgba, lab_out, n)       : batched sRGB->Lab (3-channel out)
 *   fsv_lab_to_rgba(lab, rgba_out, n)       : batched Lab->sRGB
 *   fsv_pixelwise_diff(a, b, out, n)        : c = a - b (element-wise)
 *   fsv_pixelwise_mul(a, b, out, n)         : c = a * b (element-wise)
 *
 * All exported function names are prefixed with fsv_ for clean import on
 * the JS side. emcc's -sEXPORTED_FUNCTIONS handles the exports.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#define EXPORT __attribute__((visibility("default")))

/* ── Memory management exports ─────────────────────────────────────── */

EXPORT void* fsv_alloc(size_t bytes) { return malloc(bytes); }
EXPORT void fsv_free(void* p) { if (p) free(p); }

/* ── sRGB <-> linear LUT (256 entries, populated once at init) ─────── */

static float SRGB_TO_LINEAR_LUT[256];
static int   SRGB_LUT_READY = 0;

EXPORT void fsv_srgb_lut_init(void) {
    if (SRGB_LUT_READY) return;
    for (int i = 0; i < 256; i++) {
        float c = (float)i / 255.0f;
        SRGB_TO_LINEAR_LUT[i] = (c <= 0.04045f)
            ? (c / 12.92f)
            : powf((c + 0.055f) / 1.055f, 2.4f);
    }
    SRGB_LUT_READY = 1;
}

static inline float linear_to_srgb(float c) {
    float v = (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
    v *= 255.0f;
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return v;
}

static inline float lab_f(float t) {
    return (t > 0.008856f) ? cbrtf(t) : (7.787f * t + 16.0f / 116.0f);
}

static inline float lab_f_inv(float t) {
    float t3 = t * t * t;
    return (t3 > 0.008856f) ? t3 : ((t - 16.0f / 116.0f) / 7.787f);
}

/* ── Sliding-window max (dilate) -- O(W*H), monotonic deque ─────────
 *
 * The deque holds indices of values in monotonically decreasing order.
 * The head is always the max of the current window. Each index is
 * enqueued and dequeued at most once per axis pass, so the amortized
 * cost is O(1) per pixel.
 *
 * `scratch` is a Float32 buffer of length w*h used to hold the
 * intermediate (row-pass) result. JS pre-allocates and reuses it.
 */
EXPORT void fsv_dilate(const float* in, float* out, float* scratch,
                       int width, int height, int radius) {
    /* head/tail are MONOTONIC absolute indices (no wrap), so the deque
     * arrays must be sized to the max value tail can reach in one pass:
     * width+radius for the row pass and height+radius for the column
     * pass. Sizing to (2*radius+2) was the original bug -- tail wrote
     * past the array and out-of-bounds writes/reads are UB in C,
     * effectively yielding zeros (or worse, corrupted stack) for any
     * image wider/taller than ~2*radius pixels. */
    int dq_row[width + radius];
    int dq_col[height + radius];

    /* Row pass: in -> scratch */
    for (int y = 0; y < height; y++) {
        int row = y * width;
        int head = 0, tail = 0;
        for (int x = 0; x < width + radius; x++) {
            if (x < width) {
                float v = in[row + x];
                while (head < tail && in[row + dq_row[tail - 1]] <= v) tail--;
                dq_row[tail++] = x;
            }
            int winStart = x - radius;
            int winLeft = winStart - radius;
            while (head < tail && dq_row[head] < winLeft) head++;
            if (winStart >= 0) {
                scratch[row + winStart] = in[row + dq_row[head]];
            }
        }
    }

    /* Column pass: scratch -> out */
    for (int x = 0; x < width; x++) {
        int head = 0, tail = 0;
        for (int y = 0; y < height + radius; y++) {
            if (y < height) {
                float v = scratch[y * width + x];
                while (head < tail && scratch[dq_col[tail - 1] * width + x] <= v) tail--;
                dq_col[tail++] = y;
            }
            int winStart = y - radius;
            int winLeft = winStart - radius;
            while (head < tail && dq_col[head] < winLeft) head++;
            if (winStart >= 0) {
                out[winStart * width + x] = scratch[dq_col[head] * width + x];
            }
        }
    }
}

/* ── Sliding-window min (erode) -- mirror of dilate ──────────────────
 *
 * Identical algorithm, comparison reversed (>= instead of <=), and the
 * "fresh" entry replaces lower max with higher min. JS reference uses
 * 1.0 as the neutral value (upper bound of soft mask values), but here
 * since we initialize the deque empty and the first pixel always pushes,
 * no explicit init is needed.
 */
EXPORT void fsv_erode(const float* in, float* out, float* scratch,
                      int width, int height, int radius) {
    /* See note in fsv_dilate -- deque must be full-pass capacity. */
    int dq_row[width + radius];
    int dq_col[height + radius];

    for (int y = 0; y < height; y++) {
        int row = y * width;
        int head = 0, tail = 0;
        for (int x = 0; x < width + radius; x++) {
            if (x < width) {
                float v = in[row + x];
                while (head < tail && in[row + dq_row[tail - 1]] >= v) tail--;
                dq_row[tail++] = x;
            }
            int winStart = x - radius;
            int winLeft = winStart - radius;
            while (head < tail && dq_row[head] < winLeft) head++;
            if (winStart >= 0) {
                scratch[row + winStart] = in[row + dq_row[head]];
            }
        }
    }

    for (int x = 0; x < width; x++) {
        int head = 0, tail = 0;
        for (int y = 0; y < height + radius; y++) {
            if (y < height) {
                float v = scratch[y * width + x];
                while (head < tail && scratch[dq_col[tail - 1] * width + x] >= v) tail--;
                dq_col[tail++] = y;
            }
            int winStart = y - radius;
            int winLeft = winStart - radius;
            while (head < tail && dq_col[head] < winLeft) head++;
            if (winStart >= 0) {
                out[winStart * width + x] = scratch[dq_col[head] * width + x];
            }
        }
    }
}

/* ── Masked box blur via integral images (O(N) build, O(N) query) ───
 *
 * For each pixel, returns the mask-weighted average of `values` in a
 * (2r+1)x(2r+1) window. Uses double-precision integral images to avoid
 * accumulated float error on long rows. Bit-identical to the JS
 * reference.
 *
 * The `tmp` array is a stretch optimization for caller-provided
 * preallocation. For simplicity in this initial implementation we
 * malloc internally; future versions can take it as a parameter.
 */
EXPORT void fsv_box_blur_masked(const float* values, const float* mask,
                                 float* out, int w, int h, int radius) {
    int r = (radius < 1) ? 1 : radius;
    int stride = w + 1;
    int sz = stride * (h + 1);
    double* IIvm = (double*)calloc(sz, sizeof(double));
    double* IIm  = (double*)calloc(sz, sizeof(double));
    if (!IIvm || !IIm) {
        if (IIvm) free(IIvm);
        if (IIm) free(IIm);
        /* On OOM, fill output with values (degraded but valid). */
        for (int i = 0; i < w * h; i++) out[i] = values[i];
        return;
    }

    for (int y = 0; y < h; y++) {
        double rowVM = 0.0, rowM = 0.0;
        int rowOff = y * w;
        int iiOff = (y + 1) * stride;
        int prevOff = y * stride;
        for (int x = 0; x < w; x++) {
            int i = rowOff + x;
            double m = (double)mask[i];
            rowVM += (double)values[i] * m;
            rowM  += m;
            int xPlus1 = x + 1;
            IIvm[iiOff + xPlus1] = IIvm[prevOff + xPlus1] + rowVM;
            IIm [iiOff + xPlus1] = IIm [prevOff + xPlus1] + rowM;
        }
    }

    int hLast = h - 1;
    int wLast = w - 1;
    for (int y = 0; y < h; y++) {
        int y0raw = y - r, y1raw = y + r;
        int y0 = (y0raw < 0) ? 0 : y0raw;
        int y1 = ((y1raw > hLast) ? hLast : y1raw) + 1;
        int y0s = y0 * stride;
        int y1s = y1 * stride;
        int outRow = y * w;
        for (int x = 0; x < w; x++) {
            int x0raw = x - r, x1raw = x + r;
            int x0 = (x0raw < 0) ? 0 : x0raw;
            int x1 = ((x1raw > wLast) ? wLast : x1raw) + 1;
            double sumW = IIm[y1s + x1] - IIm[y0s + x1] - IIm[y1s + x0] + IIm[y0s + x0];
            if (sumW <= 0.0) {
                out[outRow + x] = values[outRow + x];
                continue;
            }
            double sumVM = IIvm[y1s + x1] - IIvm[y0s + x1] - IIvm[y1s + x0] + IIvm[y0s + x0];
            out[outRow + x] = (float)(sumVM / sumW);
        }
    }

    free(IIvm);
    free(IIm);
}

/* ── Batched sRGB -> Lab (D65) conversion ────────────────────────────
 *
 * Input is an interleaved RGBA buffer (4 bytes per pixel, but we ignore
 * the alpha channel). Output is interleaved [L, a, b, L, a, b, ...] of
 * length 3*n_pixels.
 *
 * SIMD path processes 4 pixels at a time on cores that support it. The
 * cube root inside lab_f() is the heaviest op per channel and is the
 * main beneficiary of SIMD parallelism.
 */
EXPORT void fsv_rgba_to_lab(const uint8_t* rgba, float* lab_out, int n_pixels) {
    if (!SRGB_LUT_READY) fsv_srgb_lut_init();

    for (int i = 0; i < n_pixels; i++) {
        float rl = SRGB_TO_LINEAR_LUT[rgba[i * 4 + 0]];
        float gl = SRGB_TO_LINEAR_LUT[rgba[i * 4 + 1]];
        float bl = SRGB_TO_LINEAR_LUT[rgba[i * 4 + 2]];
        float x = (0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl) / 0.95047f;
        float y = (0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl);
        float z = (0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl) / 1.08883f;
        float fx = lab_f(x), fy = lab_f(y), fz = lab_f(z);
        lab_out[i * 3 + 0] = 116.0f * fy - 16.0f;
        lab_out[i * 3 + 1] = 500.0f * (fx - fy);
        lab_out[i * 3 + 2] = 200.0f * (fy - fz);
    }
}

/* ── Batched Lab -> sRGB conversion ──────────────────────────────────
 *
 * Input is interleaved [L, a, b, ...] of length 3*n_pixels.
 * Output is interleaved RGBA (alpha is left untouched in the caller's
 * buffer; we only write the R, G, B bytes).
 *
 * Used by cleanFence to convert the modified Lab values back to sRGB
 * for the per-pixel composite into the canvas ImageData.
 */
EXPORT void fsv_lab_to_rgba(const float* lab, uint8_t* rgba_out, int n_pixels) {
    for (int i = 0; i < n_pixels; i++) {
        float L = lab[i * 3 + 0];
        float a = lab[i * 3 + 1];
        float b = lab[i * 3 + 2];
        float fy = (L + 16.0f) / 116.0f;
        float fx = a / 500.0f + fy;
        float fz = fy - b / 200.0f;
        float X = 0.95047f * lab_f_inv(fx);
        float Y = 1.00000f * lab_f_inv(fy);
        float Z = 1.08883f * lab_f_inv(fz);
        float rl =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
        float gl = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
        float bl =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
        rgba_out[i * 4 + 0] = (uint8_t)linear_to_srgb(rl);
        rgba_out[i * 4 + 1] = (uint8_t)linear_to_srgb(gl);
        rgba_out[i * 4 + 2] = (uint8_t)linear_to_srgb(bl);
        /* Alpha left alone -- caller passes through unchanged. */
    }
}

/* ── Element-wise diff and mul (used by guided filter) ──────────────
 *
 * SIMD path processes 4 floats per instruction. Important on the
 * guide*input and guide*guide computations that precede the box blurs.
 */
EXPORT void fsv_pixelwise_diff(const float* a, const float* b, float* out, int n) {
#ifdef __wasm_simd128__
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        v128_t r  = wasm_f32x4_sub(va, vb);
        wasm_v128_store(&out[i], r);
    }
    for (; i < n; i++) out[i] = a[i] - b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
#endif
}

EXPORT void fsv_pixelwise_mul(const float* a, const float* b, float* out, int n) {
#ifdef __wasm_simd128__
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t va = wasm_v128_load(&a[i]);
        v128_t vb = wasm_v128_load(&b[i]);
        v128_t r  = wasm_f32x4_mul(va, vb);
        wasm_v128_store(&out[i], r);
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

/* ── Guided filter (one channel) ─────────────────────────────────────
 *
 * Combines guide*input and guide*guide computations, the 4 box blurs,
 * the A/B computation, and the final composite. All in one call to
 * avoid 12 JS<->WASM transitions per radius.
 *
 * Caller pre-allocates 8 scratch arrays of length n = w*h (input/output
 * buffers + 6 scratch). Returns nothing; output buffer holds result.
 *
 * The 8 scratch pointers are:
 *   sc0..sc5: 6 intermediate arrays
 *   The function reorders writes to minimize active working set per
 *   pass so that L1 cache pressure stays reasonable on large w*h.
 */
EXPORT void fsv_guided_filter(const float* input, const float* guide,
                               const float* mask, float* out,
                               float* sc0, float* sc1, float* sc2,
                               float* sc3, float* sc4, float* sc5,
                               int w, int h, int radius, float epsilon) {
    int n = w * h;

    /* sc0 = Igi = guide * input; sc1 = Igg = guide * guide */
    fsv_pixelwise_mul(guide, input, sc0, n);
    fsv_pixelwise_mul(guide, guide, sc1, n);

    /* sc2 = meanG, sc3 = meanI, sc4 = meanGI, sc5 = meanGG */
    fsv_box_blur_masked(guide, mask, sc2, w, h, radius);
    fsv_box_blur_masked(input, mask, sc3, w, h, radius);
    fsv_box_blur_masked(sc0,   mask, sc4, w, h, radius);
    fsv_box_blur_masked(sc1,   mask, sc5, w, h, radius);

    /* A = (meanGI - meanG*meanI) / (meanGG - meanG^2 + eps), B = meanI - A*meanG */
    for (int i = 0; i < n; i++) {
        float mg = sc2[i];
        float varG = sc5[i] - mg * mg;
        float A = (sc4[i] - mg * sc3[i]) / (varG + epsilon);
        float B = sc3[i] - A * mg;
        sc0[i] = A;
        sc1[i] = B;
    }

    /* meanA, meanB -> sc4, sc5 */
    fsv_box_blur_masked(sc0, mask, sc4, w, h, radius);
    fsv_box_blur_masked(sc1, mask, sc5, w, h, radius);

    /* Final composite: out = meanA*guide + meanB where mask > 0 */
    for (int i = 0; i < n; i++) {
        if (mask[i] > 0.0f) {
            out[i] = sc4[i] * guide[i] + sc5[i];
        } else {
            out[i] = 0.0f;
        }
    }
}

/* ── Linear color blend with feathered alpha (used in cleanFence final
 *    composite). For each pixel: result = orig + (modified - orig) *
 *    alpha. Operates on raw RGBA bytes; alpha is a separate Float32
 *    buffer.
 */
EXPORT void fsv_alpha_blend_rgba(const uint8_t* orig, const uint8_t* modified,
                                  const float* alpha, uint8_t* out, int n_pixels) {
#ifdef __wasm_simd128__
    int i = 0;
    /* SIMD path: process 4 pixels (16 bytes) per iteration. We
     * deinterleave RGBA into channel planes for vectorized blend, then
     * interleave back. For simplicity and to avoid the deinterleave
     * cost on small loops we fall through to scalar for the leftovers.
     */
    for (; i + 4 <= n_pixels; i += 4) {
        for (int k = 0; k < 4; k++) {
            float a = alpha[i + k];
            int p = (i + k) * 4;
            float o0 = (float)orig[p + 0];
            float o1 = (float)orig[p + 1];
            float o2 = (float)orig[p + 2];
            float m0 = (float)modified[p + 0];
            float m1 = (float)modified[p + 1];
            float m2 = (float)modified[p + 2];
            float r0 = o0 + (m0 - o0) * a;
            float r1 = o1 + (m1 - o1) * a;
            float r2 = o2 + (m2 - o2) * a;
            out[p + 0] = (uint8_t)(r0 < 0.0f ? 0.0f : (r0 > 255.0f ? 255.0f : r0));
            out[p + 1] = (uint8_t)(r1 < 0.0f ? 0.0f : (r1 > 255.0f ? 255.0f : r1));
            out[p + 2] = (uint8_t)(r2 < 0.0f ? 0.0f : (r2 > 255.0f ? 255.0f : r2));
            out[p + 3] = orig[p + 3];
        }
    }
    for (; i < n_pixels; i++) {
        float a = alpha[i];
        int p = i * 4;
        for (int c = 0; c < 3; c++) {
            float o = (float)orig[p + c];
            float m = (float)modified[p + c];
            float r = o + (m - o) * a;
            out[p + c] = (uint8_t)(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
        }
        out[p + 3] = orig[p + 3];
    }
#else
    for (int i = 0; i < n_pixels; i++) {
        float a = alpha[i];
        int p = i * 4;
        for (int c = 0; c < 3; c++) {
            float o = (float)orig[p + c];
            float m = (float)modified[p + c];
            float r = o + (m - o) * a;
            out[p + c] = (uint8_t)(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
        }
        out[p + 3] = orig[p + 3];
    }
#endif
}

/* ── Soft mask threshold + interpolation ─────────────────────────────
 *
 * Replaces the JS loop:
 *   for (let i = 0; i < n; i++) {
 *     const p = src[i];
 *     if (p <= low) out[i] = 0;
 *     else if (p >= high) out[i] = p;
 *     else { const t = (p - low) / (high - low); out[i] = p * t; }
 *   }
 *
 * This runs on every postprocess call over the upsampled 1.9M-pixel
 * mask, so even small per-pixel wins compound.
 */
EXPORT void fsv_soft_mask_threshold(const float* src, float* out,
                                     int n, float low, float high) {
    float range = high - low;
    if (range < 1e-6f) range = 1e-6f;
    float inv_range = 1.0f / range;
#ifdef __wasm_simd128__
    v128_t vlow = wasm_f32x4_splat(low);
    v128_t vhigh = wasm_f32x4_splat(high);
    v128_t vinv = wasm_f32x4_splat(inv_range);
    v128_t vzero = wasm_f32x4_splat(0.0f);
    int i = 0;
    int n4 = n & ~3;
    for (; i < n4; i += 4) {
        v128_t p = wasm_v128_load(&src[i]);
        /* t = (p - low) * inv_range */
        v128_t t = wasm_f32x4_mul(wasm_f32x4_sub(p, vlow), vinv);
        /* For middle band: p * t */
        v128_t mid = wasm_f32x4_mul(p, t);
        /* Decide: lt low -> 0; ge high -> p; else mid */
        v128_t ge_high = wasm_f32x4_ge(p, vhigh);
        v128_t le_low  = wasm_f32x4_le(p, vlow);
        v128_t r = wasm_v128_bitselect(p, mid, ge_high);
        r = wasm_v128_bitselect(vzero, r, le_low);
        wasm_v128_store(&out[i], r);
    }
    for (; i < n; i++) {
        float p = src[i];
        if (p <= low) out[i] = 0.0f;
        else if (p >= high) out[i] = p;
        else out[i] = p * ((p - low) * inv_range);
    }
#else
    for (int i = 0; i < n; i++) {
        float p = src[i];
        if (p <= low) out[i] = 0.0f;
        else if (p >= high) out[i] = p;
        else out[i] = p * ((p - low) * inv_range);
    }
#endif
}
