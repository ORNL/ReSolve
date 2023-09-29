#pragma once
#include <string>
#include <resolve/Common.hpp>

namespace ReSolve { namespace vector {
  class VectorBase 
  {
    public:
      VectorBase(index_type n);
      ~VectorBase();

      index_type getSize();
      index_type getCurrentSize();

      int setCurrentSize(index_type new_n_current);

    protected:
      index_type n_; ///< size
      index_type n_current_; // if vectors dynamically change size, "current n_" keeps track of this. Needed for some solver implementations. 
  };
}} // namespace ReSolve::vector
