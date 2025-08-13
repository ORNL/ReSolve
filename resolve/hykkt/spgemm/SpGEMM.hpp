#pragma once
#include "SpGEMMImpl.hpp"

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>

namespace ReSolve {
  using real_type = ReSolve::real_type;

  namespace hykkt {
    class SpGEMM {
      public:
        // computes E = alpha * A * B + beta * D
        SpGEMM(memory::MemorySpace memspace, real_type alpha, real_type beta);
        ~SpGEMM();

        void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
        void addSumMatrix(matrix::Csr* D);
        void addResultMatrix(matrix::Csr* E);

        void compute();

      private:
        memory::MemorySpace memspace_;

        SpGEMMImpl* impl_;
    };
  }
}