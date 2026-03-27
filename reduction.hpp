#include <stddef.h>

#if defined(__x86_64__)
    #define AVX
#elif defined(__aarch64__) || defined(__arm__)
    #if defined(__ARM_FEATURE_SVE)
        #define SVE
    #endif
#endif

#ifdef AVX
double reduction_f64(const double* data, size_t n);
double reduction_align_f64(const double* data, size_t n);
double reduction_align_64n_f64(const double* data, size_t n);
#endif

#ifdef SVE
double reduction_f64(const double* data, size_t n);
double reduction_align_f64(const double* data, size_t n);
double reduction_align_64n_f64(const double* data, size_t n);
#endif
