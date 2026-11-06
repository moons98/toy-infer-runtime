// Print the current GPU's specs.
// Used as reference values when interpreting measurements (e.g. % of peak bandwidth).

#include <cstdio>

#include "tir/cuda_check.h"

int main() {
    int device = 0;
    TIR_CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    TIR_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int sm_clock_khz = 0;
    int mem_clock_khz = 0;
    TIR_CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device));
    TIR_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));

    // DDR-type memory transfers twice per clock, hence the factor of 2
    double bandwidth_gbs = 2.0 * mem_clock_khz * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    std::printf("Device %d: %s\n", device, prop.name);
    std::printf("  Compute capability      : %d.%d\n", prop.major, prop.minor);
    std::printf("  SM count                : %d\n", prop.multiProcessorCount);
    std::printf("  Max threads per block   : %d\n", prop.maxThreadsPerBlock);
    std::printf("  Max threads per SM      : %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("  Warp size               : %d\n", prop.warpSize);
    std::printf("  Shared memory per block : %zu KB\n", prop.sharedMemPerBlock / 1024);
    std::printf("  Global memory           : %.1f GB\n", prop.totalGlobalMem / 1e9);
    std::printf("  SM clock                : %.0f MHz\n", sm_clock_khz / 1e3);
    std::printf("  Memory bus width        : %d bit\n", prop.memoryBusWidth);
    std::printf("  Peak memory bandwidth   : %.0f GB/s (theoretical)\n", bandwidth_gbs);
    return 0;
}
