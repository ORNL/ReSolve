#pragma once

namespace ReSolve {
  namespace hykkt {
    class SpGEMMImpl {
      public:
        SpGEMMImpl() = default;
        ~SpGEMMImpl() = default;

        virtual void addProductMatrices(matrix::Csr* A, matrix::Csr* B) = 0;
        virtual void addSumMatrix(matrix::Csr* D) = 0;
        virtual void addResultMatrix(matrix::Csr* E) = 0;

        virtual void compute() = 0;
    };
  }
}