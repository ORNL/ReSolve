#pragma once

#include <cusparse.h>

#include "SpGEMMImpl.hpp"

namespace ReSolve {
  using real_type = ReSolve::real_type;

  namespace hykkt {
    class SpGEMMCuda : public SpGEMMImpl {
      public:
        SpGEMMCuda(real_type alpha, real_type beta);
        ~SpGEMMCuda();

        void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
        void addSumMatrix(matrix::Csr* D);
        void addResultMatrix(matrix::Csr** E_ptr);

        void compute();

      private:
        MemoryHandler mem_;

        real_type alpha_;
        real_type beta_;

        matrix::Csr* D_ = nullptr;
        matrix::Csr** E_ptr_ = nullptr;

        cusparseHandle_t handle_ = nullptr;
        cusparseSpGEMMDescr_t spgemm_desc_ = nullptr;

        size_t buffer_size_4_ = 0;
        size_t buffer_size_5_ = 0;
        void* buffer_4_ = nullptr;
        void* buffer_5_ = nullptr;

        size_t buffer_add_size_ = 0;
        void* buffer_add_ = nullptr;

        cusparseSpMatDescr_t A_descr_ = nullptr;
        cusparseSpMatDescr_t B_descr_ = nullptr;
        cusparseSpMatDescr_t C_descr_ = nullptr;

        cusparseMatDescr_t D_descr_ = nullptr;

        int64_t n_ = 0;
        int64_t C_nnz_ = 0;

        index_type* C_row_ptr_ = nullptr;
        index_type* C_col_ind_ = nullptr;
        real_type* C_val_ = nullptr;

        int E_nnz_ = 0;

        index_type* E_row_ptr_ = nullptr; 
        index_type* E_col_ind_ = nullptr;
        real_type* E_val_ = nullptr;

        cusparseSpMatDescr_t convertToCusparseType(matrix::Csr* A);
    };
  }
}