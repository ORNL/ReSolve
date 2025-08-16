#pragma once

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/matrix/Csr.hpp>

namespace ReSolve {
  using real_type = ReSolve::real_type;

  namespace hykkt {
    class SpGEMMImpl {
      public:
        SpGEMMImpl() = default;
        virtual ~SpGEMMImpl() = default;

        virtual void addProductMatrices(matrix::Csr* A, matrix::Csr* B) = 0;
        virtual void addSumMatrix(matrix::Csr* D) = 0;
        virtual void addResultMatrix(matrix::Csr** E_ptr) = 0;

        virtual void compute() = 0;
    };
  }
}