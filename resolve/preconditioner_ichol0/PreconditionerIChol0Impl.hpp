/**
 * @file PreconditionerIChol0Impl.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Abstract interface for Cholesky Solver implementations
 */

#pragma once
#include <resolve/Common.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

// ANDREW TODO: remove memspace arguments
namespace ReSolve
{
  class PreconditionerIChol0Impl
  {
  public:
    PreconditionerIChol0Impl()          = default;
    virtual ~PreconditionerIChol0Impl() = default;

    virtual int setup(matrix::Csr* A) = 0;
    virtual int apply(vector::Vector* rhs, vector::Vector* x) = 0;
    virtual void setNumRhs(index_type num_rhs) = 0;
    };
} // namespace ReSolve
