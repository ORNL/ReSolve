#include <cstring>
#include <cuda_runtime.h>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorKernels.hpp>

namespace ReSolve { namespace vector {

  VectorBase::Vector(index_type n):
    n_{n}
  {
    n_current_ = n_;
  }

  VectorBase::~Vector()
  {
  }

  index_type VectorBase::getSize()
  {
    return n_;
  }

  index_type VectorBase::getCurrentSize()
  {
    return n_current_;
  }

  int VectorBase::setCurrentSize(int new_n_current)
  {
    if (new_n_current > n_) {
      return -1;
    } else {
      n_current_ = new_n_current;
      return 0;
    }
  }

}} // namespace ReSolve::vector
