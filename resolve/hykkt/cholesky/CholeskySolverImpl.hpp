#pragma once
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverImpl
    {
    public:
      CholeskySolverImpl()  = default;
      ~CholeskySolverImpl() = default;

      virtual void addMatrixInfo(matrix::Csr* A)               = 0;
      virtual void symbolicAnalysis()                          = 0;
      virtual void numericalFactorization(real_type tol)       = 0;
      virtual void solve(vector::Vector* x, vector::Vector* b) = 0;
    };
  } // namespace hykkt
} // namespace ReSolve
