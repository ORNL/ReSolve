#pragma once

#ifdef RESOLVE_USE_PROFILING

#ifdef RESOLVE_USE_GPU
#ifdef RESOLVE_USE_HIP
#include <rocprofiler-sdk-roctx/roctx.h>
#define RESOLVE_RANGE_PUSH(x) roctxRangePush(x)
#define RESOLVE_RANGE_POP(x) \
  roctxRangePop();           \
  roctxMarkA(x)
#endif // RESOLVE_USE_HIP

#ifdef RESOLVE_USE_CUDA
#ifdef RESOLVE_USE_NVTX3
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif // RESOLVE_USE_NVTX3
#define RESOLVE_RANGE_PUSH(x) nvtxRangePush(x)
#define RESOLVE_RANGE_POP(x) \
  nvtxRangePop();            \
  nvtxMarkA(x)
#endif // RESOLVE_USE_CUDA

#else

// Not using GPU
#define RESOLVE_RANGE_PUSH(x)
#define RESOLVE_RANGE_POP(x)

#endif // RESOLVE_USE_GPU

#else

// Not using profiling
#define RESOLVE_RANGE_PUSH(x)
#define RESOLVE_RANGE_POP(x)

#endif // RESOLVE_USE_PROFILING
