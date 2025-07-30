#pragma once
#include "SpGEMMImpl.hpp"

#include <resolve/MemoryUtils.hpp>
#include <resolve/matrix/Csr.hpp>

namespace ReSolve {
  namespace hykkt {
    class SpGEMM {
      public:
        SpGEMM(memory::MemorySpace memspace, real_type alpha, real_type beta);
        ~SpGEMM();

        void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
        void addSumMatrix(matrix::Csr* D);
        void addResultMatrix(matrix::Csr* E);

        void compute();

      private:
        memory::MemorySpace memspace_;

        matrix::Csr* A;
        matrix::Csr* B;
        matrix::Csr* D;
        matrix::Csr* E;

        SpGEMMImpl* impl_;
    };
  }
}