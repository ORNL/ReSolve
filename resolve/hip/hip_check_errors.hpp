/**
 * @file hip_check_errors.hpp
 * 
 * Contains macro to get error code from CUDA functions and to stream
 * appropriate error output to Re::Solve's logger.
 * 
 * @author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>
 * @author Slaven Peles <peless@ornl.gov>
 */
#pragma once

#include <resolve/utilities/logger/Logger.hpp>

template <typename T>
int  check(T result, 
           char const *const func, 
           const char *const file,
           int const line) 
{
  if (result) {
    ReSolve::io::Logger::error() << "HIP error in function "
                                 << func << " at " << file << ":" << line 
                                 << ", error# " << result << "\n";
    return -1;
  }
  return 0;
}
// #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define checkHipErrors(val) val
