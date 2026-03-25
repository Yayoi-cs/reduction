#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
    #define AVX
#elif defined(__aarch64__) || defined(__arm__)
    #if defined(__ARM_FEATURE_SVE)
        #define SVE
    #endif
#endif

#ifdef AVX

#include <immintrin.h>

double reduction_f64(const double* data, size_t n) {
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();
    __m512d acc4 = _mm512_setzero_pd();
    __m512d acc5 = _mm512_setzero_pd();
    __m512d acc6 = _mm512_setzero_pd();
    __m512d acc7 = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(data + i));
        acc1 = _mm512_add_pd(acc1, _mm512_loadu_pd(data + i + 8));
        acc2 = _mm512_add_pd(acc2, _mm512_loadu_pd(data + i + 16));
        acc3 = _mm512_add_pd(acc3, _mm512_loadu_pd(data + i + 24));
        acc4 = _mm512_add_pd(acc4, _mm512_loadu_pd(data + i + 32));
        acc5 = _mm512_add_pd(acc5, _mm512_loadu_pd(data + i + 40));
        acc6 = _mm512_add_pd(acc6, _mm512_loadu_pd(data + i + 48));
        acc7 = _mm512_add_pd(acc7, _mm512_loadu_pd(data + i + 56));
    }

    acc0 = _mm512_add_pd(acc0, acc4);
    acc1 = _mm512_add_pd(acc1, acc5);
    acc2 = _mm512_add_pd(acc2, acc6);
    acc3 = _mm512_add_pd(acc3, acc7);
    acc0 = _mm512_add_pd(acc0, acc2);
    acc1 = _mm512_add_pd(acc1, acc3);
    acc0 = _mm512_add_pd(acc0, acc1);

    for (; i + 8 <= n; i += 8)
        acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(data + i));

    if (i < n) {
        __mmask8 tail = (__mmask8)((1u << (n - i)) - 1);
        acc0 = _mm512_add_pd(acc0, _mm512_maskz_loadu_pd(tail, data + i));
    }

    return _mm512_reduce_add_pd(acc0);
}

double reduction_align_f64(const double* __restrict__ data, size_t n) {
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();
    __m512d acc4 = _mm512_setzero_pd();
    __m512d acc5 = _mm512_setzero_pd();
    __m512d acc6 = _mm512_setzero_pd();
    __m512d acc7 = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_add_pd(acc0, _mm512_load_pd(data + i));
        acc1 = _mm512_add_pd(acc1, _mm512_load_pd(data + i + 8));
        acc2 = _mm512_add_pd(acc2, _mm512_load_pd(data + i + 16));
        acc3 = _mm512_add_pd(acc3, _mm512_load_pd(data + i + 24));
        acc4 = _mm512_add_pd(acc4, _mm512_load_pd(data + i + 32));
        acc5 = _mm512_add_pd(acc5, _mm512_load_pd(data + i + 40));
        acc6 = _mm512_add_pd(acc6, _mm512_load_pd(data + i + 48));
        acc7 = _mm512_add_pd(acc7, _mm512_load_pd(data + i + 56));
    }

    acc0 = _mm512_add_pd(acc0, acc4);
    acc1 = _mm512_add_pd(acc1, acc5);
    acc2 = _mm512_add_pd(acc2, acc6);
    acc3 = _mm512_add_pd(acc3, acc7);
    acc0 = _mm512_add_pd(acc0, acc2);
    acc1 = _mm512_add_pd(acc1, acc3);
    acc0 = _mm512_add_pd(acc0, acc1);

    for (; i + 8 <= n; i += 8)
        acc0 = _mm512_add_pd(acc0, _mm512_load_pd(data + i));

    if (i < n) {
        __mmask8 tail = (__mmask8)((1u << (n - i)) - 1);
        acc0 = _mm512_add_pd(acc0, _mm512_maskz_load_pd(tail, data + i));
    }

    return _mm512_reduce_add_pd(acc0);
}

double reduction_align_8n_f64(const double* __restrict__ data, size_t n) {
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = _mm512_setzero_pd();
    __m512d acc3 = _mm512_setzero_pd();
    __m512d acc4 = _mm512_setzero_pd();
    __m512d acc5 = _mm512_setzero_pd();
    __m512d acc6 = _mm512_setzero_pd();
    __m512d acc7 = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_add_pd(acc0, _mm512_load_pd(data + i));
        acc1 = _mm512_add_pd(acc1, _mm512_load_pd(data + i + 8));
        acc2 = _mm512_add_pd(acc2, _mm512_load_pd(data + i + 16));
        acc3 = _mm512_add_pd(acc3, _mm512_load_pd(data + i + 24));
        acc4 = _mm512_add_pd(acc4, _mm512_load_pd(data + i + 32));
        acc5 = _mm512_add_pd(acc5, _mm512_load_pd(data + i + 40));
        acc6 = _mm512_add_pd(acc6, _mm512_load_pd(data + i + 48));
        acc7 = _mm512_add_pd(acc7, _mm512_load_pd(data + i + 56));
    }

    acc0 = _mm512_add_pd(acc0, acc4);
    acc1 = _mm512_add_pd(acc1, acc5);
    acc2 = _mm512_add_pd(acc2, acc6);
    acc3 = _mm512_add_pd(acc3, acc7);
    acc0 = _mm512_add_pd(acc0, acc2);
    acc1 = _mm512_add_pd(acc1, acc3);
    acc0 = _mm512_add_pd(acc0, acc1);

    for (; i < n; i += 8)
        acc0 = _mm512_add_pd(acc0, _mm512_load_pd(data + i));

    return _mm512_reduce_add_pd(acc0);
}

#endif

#ifdef SVE

#include <arm_sve.h>

double reduction_f64(const double* data, size_t n) {
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    svfloat64_t acc2 = svdup_f64(0.0);
    svfloat64_t acc3 = svdup_f64(0.0);
    svfloat64_t acc4 = svdup_f64(0.0);
    svfloat64_t acc5 = svdup_f64(0.0);
    svfloat64_t acc6 = svdup_f64(0.0);
    svfloat64_t acc7 = svdup_f64(0.0);

    const int64_t vl      = (int64_t)svcntd();
    const int64_t stride8 = vl * 8;
    const svbool_t ptrue   = svptrue_b64();
    const int64_t sn       = (int64_t)n;
    int64_t i = 0;

    for (; i + stride8 <= sn; i += stride8) {
        acc0 = svadd_f64_x(ptrue, acc0, svld1_f64(ptrue, data + i));
        acc1 = svadd_f64_x(ptrue, acc1, svld1_f64(ptrue, data + i + vl));
        acc2 = svadd_f64_x(ptrue, acc2, svld1_f64(ptrue, data + i + vl*2));
        acc3 = svadd_f64_x(ptrue, acc3, svld1_f64(ptrue, data + i + vl*3));
        acc4 = svadd_f64_x(ptrue, acc4, svld1_f64(ptrue, data + i + vl*4));
        acc5 = svadd_f64_x(ptrue, acc5, svld1_f64(ptrue, data + i + vl*5));
        acc6 = svadd_f64_x(ptrue, acc6, svld1_f64(ptrue, data + i + vl*6));
        acc7 = svadd_f64_x(ptrue, acc7, svld1_f64(ptrue, data + i + vl*7));
    }

    acc0 = svadd_f64_x(ptrue, acc0, acc4);
    acc1 = svadd_f64_x(ptrue, acc1, acc5);
    acc2 = svadd_f64_x(ptrue, acc2, acc6);
    acc3 = svadd_f64_x(ptrue, acc3, acc7);
    acc0 = svadd_f64_x(ptrue, acc0, acc2);
    acc1 = svadd_f64_x(ptrue, acc1, acc3);
    acc0 = svadd_f64_x(ptrue, acc0, acc1);

    for (; i + vl <= sn; i += vl)
        acc0 = svadd_f64_x(ptrue, acc0, svld1_f64(ptrue, data + i));

    svbool_t tail_pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);
    acc0 = svadd_f64_m(tail_pg, acc0, svld1_f64(tail_pg, data + i));

    return svaddv_f64(ptrue, acc0);     /* FADDV */
}

double reduction_align_f64(const double* __restrict__ data, size_t n) {
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    svfloat64_t acc2 = svdup_f64(0.0);
    svfloat64_t acc3 = svdup_f64(0.0);
    svfloat64_t acc4 = svdup_f64(0.0);
    svfloat64_t acc5 = svdup_f64(0.0);
    svfloat64_t acc6 = svdup_f64(0.0);
    svfloat64_t acc7 = svdup_f64(0.0);

    const int64_t vl      = (int64_t)svcntd();
    const int64_t stride8 = vl * 8;
    const svbool_t ptrue   = svptrue_b64();
    const int64_t sn       = (int64_t)n;
    int64_t i = 0;

    for (; i + stride8 <= sn; i += stride8) {
        svprfd(ptrue, data + i + stride8, SV_PLDL1KEEP);

        acc0 = svadd_f64_x(ptrue, acc0, svld1_f64(ptrue, data + i));
        acc1 = svadd_f64_x(ptrue, acc1, svld1_f64(ptrue, data + i + vl));
        acc2 = svadd_f64_x(ptrue, acc2, svld1_f64(ptrue, data + i + vl*2));
        acc3 = svadd_f64_x(ptrue, acc3, svld1_f64(ptrue, data + i + vl*3));
        acc4 = svadd_f64_x(ptrue, acc4, svld1_f64(ptrue, data + i + vl*4));
        acc5 = svadd_f64_x(ptrue, acc5, svld1_f64(ptrue, data + i + vl*5));
        acc6 = svadd_f64_x(ptrue, acc6, svld1_f64(ptrue, data + i + vl*6));
        acc7 = svadd_f64_x(ptrue, acc7, svld1_f64(ptrue, data + i + vl*7));
    }

    acc0 = svadd_f64_x(ptrue, acc0, acc4);
    acc1 = svadd_f64_x(ptrue, acc1, acc5);
    acc2 = svadd_f64_x(ptrue, acc2, acc6);
    acc3 = svadd_f64_x(ptrue, acc3, acc7);
    acc0 = svadd_f64_x(ptrue, acc0, acc2);
    acc1 = svadd_f64_x(ptrue, acc1, acc3);
    acc0 = svadd_f64_x(ptrue, acc0, acc1);

    for (; i + vl <= sn; i += vl)
        acc0 = svadd_f64_x(ptrue, acc0, svld1_f64(ptrue, data + i));

    svbool_t tail_pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);
    acc0 = svadd_f64_m(tail_pg, acc0, svld1_f64(tail_pg, data + i));

    return svaddv_f64(ptrue, acc0);
}

double reduction_align_8n_f64(const double* __restrict__ data, size_t n) {
    svfloat64_t acc0 = svdup_f64(0.0);
    svfloat64_t acc1 = svdup_f64(0.0);
    svfloat64_t acc2 = svdup_f64(0.0);
    svfloat64_t acc3 = svdup_f64(0.0);
    svfloat64_t acc4 = svdup_f64(0.0);
    svfloat64_t acc5 = svdup_f64(0.0);
    svfloat64_t acc6 = svdup_f64(0.0);
    svfloat64_t acc7 = svdup_f64(0.0);

    const int64_t vl      = (int64_t)svcntd();
    const int64_t stride8 = vl * 8;
    const svbool_t ptrue   = svptrue_b64();
    const int64_t sn       = (int64_t)n;
    int64_t i = 0;

    for (; i + stride8 <= sn; i += stride8) {
        svprfd(ptrue, data + i + stride8, SV_PLDL1KEEP);

        acc0 = svadd_f64_x(ptrue, acc0, svld1_f64(ptrue, data + i));
        acc1 = svadd_f64_x(ptrue, acc1, svld1_f64(ptrue, data + i + vl));
        acc2 = svadd_f64_x(ptrue, acc2, svld1_f64(ptrue, data + i + vl*2));
        acc3 = svadd_f64_x(ptrue, acc3, svld1_f64(ptrue, data + i + vl*3));
        acc4 = svadd_f64_x(ptrue, acc4, svld1_f64(ptrue, data + i + vl*4));
        acc5 = svadd_f64_x(ptrue, acc5, svld1_f64(ptrue, data + i + vl*5));
        acc6 = svadd_f64_x(ptrue, acc6, svld1_f64(ptrue, data + i + vl*6));
        acc7 = svadd_f64_x(ptrue, acc7, svld1_f64(ptrue, data + i + vl*7));
    }

    acc0 = svadd_f64_x(ptrue, acc0, acc4);
    acc1 = svadd_f64_x(ptrue, acc1, acc5);
    acc2 = svadd_f64_x(ptrue, acc2, acc6);
    acc3 = svadd_f64_x(ptrue, acc3, acc7);
    acc0 = svadd_f64_x(ptrue, acc0, acc2);
    acc1 = svadd_f64_x(ptrue, acc1, acc3);
    acc0 = svadd_f64_x(ptrue, acc0, acc1);

    for (; i < sn; i += vl) {
        svbool_t pg = svwhilelt_b64((uint64_t)i, (uint64_t)n);
        acc0 = svadd_f64_m(pg, acc0, svld1_f64(pg, data + i));
    }

    return svaddv_f64(ptrue, acc0);
}

#endif
