#pragma once

#include <cholmod.h>

#include "SpGEMMImpl.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;

  namespace hykkt
  {
    class SpGEMMCpu : public SpGEMMImpl
    {
    public:
      SpGEMMCpu(real_type alpha, real_type beta);
      ~SpGEMMCpu();

      void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
      void addSumMatrix(matrix::Csr* D);
      void addResultMatrix(matrix::Csr** E_ptr);

      void compute();

    private:
      MemoryHandler mem_;

      real_type alpha_;
      real_type beta_;

      cholmod_common Common_;

      cholmod_sparse* A_;
      cholmod_sparse* B_;
      cholmod_sparse* D_;

      matrix::Csr** E_ptr_;

      cholmod_sparse* allocateCholmodType(matrix::Csr* X);
      void            copyValuesToCholmodType(matrix::Csr* X, cholmod_sparse* X_chol);
    };
  } // namespace hykkt
} // namespace ReSolve
