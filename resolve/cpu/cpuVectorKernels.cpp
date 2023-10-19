#include <resolve/Common.hpp>
#include <resolve/vector/VectorKernels.hpp>


namespace ReSolve { namespace vector {


void set_array_const(index_type n, real_type val, real_type* arr)
{
  for(index_type i = 0; i < n; ++i) {
    arr[i] = val;
  }
}

}} // namespace ReSolve::vector