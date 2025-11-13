/**
 * @file CholeskySolver.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Cholesky decomposition solver header
 */

#pragma once
#include "CholeskySolverImpl.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolver
    {
    public:
      CholeskySolver(memory::MemorySpace memspace);
      ~CholeskySolver();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void setPivotTolerance(real_type tol);
      void numericalFactorization();
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      memory::MemorySpace memspace_;

      matrix::Csr*        A_;
      real_type           tol_ = 1e-12;
      CholeskySolverImpl* impl_;
    };
  } // namespace hykkt
} // namespace ReSolve
