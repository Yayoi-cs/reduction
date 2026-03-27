// g++ -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -o bench bench.cpp reduction.cpp

#include "reduction.hpp"
#include "benchlib/benchlib.hpp"
#include <chrono>
#include <cstdio>
#include <numeric>
#include <execution>

#define N_TIMES 0x1000000

int main(void) {
    for (size_t i = 0x40;i < 0x4000;i=i*2) {
        auto rand_ptr = benchlib::rand_buffer<double>(i);
        double* p = rand_ptr.get();
        double sum = 0.0;
        printf("i: %lx\n",i);
        sum = 0.0;
        auto r1 = benchlib::measure([&]() {
            sum += reduction_f64(p, i);
        }, N_TIMES);
        printf("reduction_f64:  %ld ns  (sum: %A)\n", r1.count(), sum);

        sum = 0.0;
        auto r2 = benchlib::measure([&]() {
            sum += reduction_align_f64(p, i);
        }, N_TIMES);
        printf("reduction_align_f64:  %ld ns  (sum: %A)\n", r2.count(), sum);

        sum = 0.0;
        auto r3 = benchlib::measure([&]() {
            sum += reduction_align_64n_f64(p, i);
        }, N_TIMES);
        printf("reduction_align_64n_f64:  %ld ns  (sum: %A)\n", r3.count(), sum);

        sum = 0.0;
        auto r4 = benchlib::measure([&]() {
            sum += std::reduce(std::execution::unseq, p, p + i);
        }, N_TIMES);
        printf("std::reduce:  %ld ns  (sum: %A)\n", r4.count(), sum);

        puts("");
    }

    return 0;
}
