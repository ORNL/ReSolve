/**
 * @file   PreconditionerIdentity.hpp
 * @author Kakeru Ueda (k.ueda.2290@m.isct.ac.jp)
 * @brief  Declaration of identity preconditioner class.
 *
 */

#pragma once

#include <resolve/MemoryUtils.hpp>
#include <resolve/Preconditioner.hpp>

namespace ReSolve
{
  /**
   * @brief Identity preconditioner.
   *
   * Applying this preconditioner computes x = rhs. It can be used to run an
   * iterative solver without changing its preconditioner-based code path.
   */
  class PreconditionerIdentity : public Preconditioner
  {
  public:
    using vector_type = vector::Vector;
    using matrix_type = matrix::Sparse;

    explicit PreconditionerIdentity(memory::MemorySpace memspace);

    int setup(matrix_type* A) override;
    int reset(matrix_type* A) override;
    int apply(vector_type* rhs, vector_type* x) override;

  private:
    memory::MemorySpace memspace_;
  };
} // namespace ReSolve
