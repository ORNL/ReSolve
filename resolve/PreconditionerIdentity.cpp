/**
 * @file   PreconditionerIdentity.cpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Implementation of identity preconditioner class.
 *
 */

#include "PreconditionerIdentity.hpp"

#include <cassert>

#include <resolve/MemoryUtils.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  /**
   * @brief Constructs an identity preconditioner for a memory space.
   *
   * @param[in] memspace Memory space used by input and output vectors
   */
  PreconditionerIdentity::PreconditionerIdentity(memory::MemorySpace memspace)
    : memspace_(memspace)
  {
  }

  /**
   * @brief Sets up the identity preconditioner.
   *
   * The identity preconditioner has no matrix-dependent state, so setup is a
   * no-op and accepts a null matrix pointer.
   *
   * @return Always returns zero.
   */
  int PreconditionerIdentity::setup(matrix_type* /* A */)
  {
    return 0;
  }

  /**
   * @brief Applies the identity preconditioner.
   *
   * Copies rhs to x in the memory space specified at construction.
   *
   * @param[in]  rhs Input vector
   * @param[out] x   Output vector
   *
   * @return Zero on success, one if the vectors are invalid or cannot be copied.
   */
  int PreconditionerIdentity::apply(vector_type* rhs, vector_type* x)
  {
    assert(rhs != nullptr && "Input vector is null!");
    assert(x != nullptr && "Output vector is null!");
    assert(rhs->getSize() == x->getSize() && "Vector sizes do not match!");
    assert(rhs->getNumVectors() == x->getNumVectors() && "Vector counts do not match!");
    assert(rhs->isAllocated(memspace_) && "Input vector data is not allocated!");
    assert(x->isAllocated(memspace_) && "Output vector data is not allocated!");

    return x->copyFromExternal(rhs, memspace_, memspace_) == 0 ? 0 : 1;
  }
} // namespace ReSolve
