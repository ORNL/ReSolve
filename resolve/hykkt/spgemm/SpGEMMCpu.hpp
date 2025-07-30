#pragma once

#include "SpGEMMImpl.hpp"

namespace ReSolve {
  namespace hykkt {
    class SPGEMMCpu : public SpGEMMImpl {
      public:
        SPGEMMCpu();
        ~SPGEMMCpu();

        void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
        void addSumMatrix(matrix::Csr* D);
        void addResultMatrix(matrix::Csr* E);

        void compute();
    };
  }
}