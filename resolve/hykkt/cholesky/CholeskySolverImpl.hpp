#pragma once
#include <resolve/Common.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve {
  namespace hykkt {
    class CholeskySolverImpl {
    public:
      CholeskySolverImpl() = default;
      ~CholeskySolverImpl() = default;

      virtual void symbolicAnalysis(matrix::Csr* A) = 0;
      virtual void numericalFactorization(matrix::Csr* A, real_type tol) = 0;
      virtual void solve(matrix::Csr* A, vector::Vector* x, vector::Vector* b) = 0;
    };
  }
}