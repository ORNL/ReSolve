#pragma once

#ifdef RESOLVE_USE_PROFILING

#ifdef RESOLVE_USE_HIP
#include <roctracer/roctx.h>
#define RESOLVE_RANGE_PUSH(x) roctxRangePush(x)
#define RESOLVE_RANGE_POP(x) 	roctxRangePop(); \
	                            roctxMarkA(x)
#endif

#else

// Not using profiling
#define RESOLVE_RANGE_PUSH(x)
#define RESOLVE_RANGE_POP(x)

#endif // RESOLVE_USE_PROFILING
