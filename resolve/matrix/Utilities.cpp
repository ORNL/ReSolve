#include <memory>

#include "Utilities.hpp"

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Sparse.hpp>

namespace ReSolve
{
  // NOTE: so the reason this just delegates to a private `expand` method on
  //       the Sparse class is threefold:
  //       1. we need to know exactly what kind of matrix we're dealing with
  //          and that's a pretty clean way of doing it (see #3)
  //       2. this was intended to operate on both dense and sparse matrices
  //          and while it does not support the latter now, it will in the
  //          future
  //       3. we'll eventually move the code that expands it back out here once
  //          the matrix interfaces have been improved with stuff that makes
  //          possible in a generic fashion but this does not yet exist
  int matrix::expand(matrix::Sparse* A)
  {
    return A->expand();
  }

  // NOTE: i don't like these too much. they're just "good enough"

  std::function<std::tuple<std::tuple<index_type, index_type, real_type>, bool>()> matrix::elements(matrix::Sparse* A)
  {
    return A->elements(memory::HOST);
  }
} // namespace ReSolve
