#g++ -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -o bench bench.cpp reduction.cpp

g++ -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm \
    -o bench bench.cpp reduction.cpp
