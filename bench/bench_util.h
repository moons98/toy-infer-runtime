#pragma once

// Measurement helpers shared by the benchmark programs.
// The timers moved to include/tir/timer.h so main can use them too; they stay available here
// under tir::bench so the benchmarks did not need to change.

#include "tir/timer.h"

namespace tir::bench {

using tir::CpuTimer;
using tir::GpuTimer;
using tir::median;

}  // namespace tir::bench
