// g++ -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -o bench bench.cpp reduction.cpp

#include "reduction.hpp"
#include "benchlib/benchlib.hpp"
#include <chrono>
#include <cstdio>
#include <numeric>
#include <execution>

#define N_OBJ 0x1000

int main(void) {
    auto rand_ptr = benchlib::rand_buffer<double>(N_OBJ);
    double* p = rand_ptr.get();
    double sum = 0.0;

    auto r1 = benchlib::measure([&]() {
        sum += reduction_align_8n_f64(p, N_OBJ);
    }, 0x10000);
    printf("hand-rolled:  %ld ns  (sum: %A)\n", r1.count(), sum);

    sum = 0.0;
    auto r2 = benchlib::measure([&]() {
        sum += std::reduce(std::execution::unseq, p, p + N_OBJ);
    }, 0x10000);
    printf("std::reduce:  %ld ns  (sum: %A)\n", r2.count(), sum);

    return 0;
}
