// g++ -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl \
//     -I${MKLROOT}/include \
//     -L${MKLROOT}/lib/intel64 \
//     -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm \
//     -o bench bench.cpp reduction.cpp

#include "reduction.hpp"
#include "benchlib/benchlib.hpp"
#include <cstdio>
#include <numeric>
#include <execution>

#include <mkl_cblas.h>
#include <mkl_vsl.h>

#define N_TIMES 0x1000000

int main(void) {
    for (size_t i = 0x40; i < 0x4000; i = i * 2) {
        auto rand_ptr = benchlib::rand_buffer<double>(i);
        double* p = rand_ptr.get();

        std::vector<double> ones(i, 1.0);

        double sum = 0.0;
        printf("i: %lx\n", i);

        sum = 0.0;
        auto r1 = benchlib::measure([&]() {
            sum += reduction_f64(p, i);
        }, N_TIMES);
        printf("reduction_f64:           %ld ns  (sum: %A)\n", r1.count(), sum);

        sum = 0.0;
        auto r2 = benchlib::measure([&]() {
            sum += reduction_align_f64(p, i);
        }, N_TIMES);
        printf("reduction_align_f64:     %ld ns  (sum: %A)\n", r2.count(), sum);

        sum = 0.0;
        auto r3 = benchlib::measure([&]() {
            sum += reduction_align_64n_f64(p, i);
        }, N_TIMES);
        printf("reduction_align_64n_f64: %ld ns  (sum: %A)\n", r3.count(), sum);

        sum = 0.0;
        auto r4 = benchlib::measure([&]() {
            sum += std::reduce(std::execution::unseq, p, p + i);
        }, N_TIMES);
        printf("std::reduce:             %ld ns  (sum: %A)\n", r4.count(), sum);

        sum = 0.0;
        auto r5 = benchlib::measure([&]() {
            sum += cblas_ddot((MKL_INT)i, p, 1, ones.data(), 1);
        }, N_TIMES);
        printf("mkl_cblas_ddot:          %ld ns  (sum: %A)\n", r5.count(), sum);

        VSLSSTaskPtr task;
        double mkl_mean = 0.0;
        double mkl_sum_result = 0.0;
        MKL_INT p_dim = 1;
        MKL_INT n_dim = (MKL_INT)i;
        vsldSSNewTask(&task, &p_dim, &n_dim, &VSL_SS_MATRIX_STORAGE_ROWS, p, nullptr, nullptr);
        vsldSSEditTask(task, VSL_SS_ED_MEAN, &mkl_mean);

        sum = 0.0;
        auto r6 = benchlib::measure([&]() {
            vsldSSCompute(task, VSL_SS_MEAN, VSL_SS_METHOD_FAST);
            sum += mkl_mean * (double)i;
        }, N_TIMES);
        printf("mkl_vsl_mean*n:          %ld ns  (sum: %A)\n", r6.count(), sum);
        vslSSDeleteTask(&task);

        puts("");
    }

    return 0;
}
