# qf-vol-surface-engine

An arbitrage-free implied volatility surface and multi-model option pricing library.

C++20 core, Python bindings. Every pricer is cross-validated against every other
pricer; every reported number is measured against a known-correct reference.

Work in progress.

## Layout

    cpp/include/vse/   header-only C++20 core
    cpp/tests/         C++ test binary
    cpp/bench/         benchmark binary
    bindings/          pybind11 module
    python/vsepy/      Python layer: chain cleaning, calibration drivers, plots
    benchmarks/        measured results, checked in

## Build

    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build
    cmake --build build
    ./build/vse_tests
