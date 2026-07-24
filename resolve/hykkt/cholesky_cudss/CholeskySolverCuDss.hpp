/**
 * @file CholeskySolverCuDss.hpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Cholesky decomposition (cuDSS implementation) solver header. This is a CUDA-only variant of CholeskySolver.
 */

#pragma once
#include "CholeskySolverCuDssCuda.hpp"
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace hykkt
  {
    class CholeskySolverCuDss
    {
    public:
      CholeskySolverCuDss(memory::MemorySpace memspace);
      ~CholeskySolverCuDss();

      void addMatrixInfo(matrix::Csr* A);
      void symbolicAnalysis();
      void setPivotTolerance(real_type tol);
      void numericalFactorization();
      void solve(vector::Vector* x, vector::Vector* b);

    private:
      memory::MemorySpace memspace_;

      matrix::Csr*             A_;
      real_type                tol_ = 1e-12;
      CholeskySolverCuDssCuda* impl_;
    };
  } // namespace hykkt
} // namespace ReSolve
