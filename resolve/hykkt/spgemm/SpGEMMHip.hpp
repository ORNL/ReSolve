#pragma once

#include <rocsparse/rocsparse.h>

#include "SpGEMMImpl.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;

  namespace hykkt
  {
    class SpGEMMHip : public SpGEMMImpl
    {
    public:
      SpGEMMHip(real_type alpha, real_type beta);
      ~SpGEMMHip();

      void addProductMatrices(matrix::Csr* A, matrix::Csr* B);
      void addSumMatrix(matrix::Csr* D);
      void addResultMatrix(matrix::Csr** E_ptr);

      void compute();

    private:
      MemoryHandler mem_;

      real_type alpha_;
      real_type beta_;

      rocsparse_handle      handle_  = nullptr;
      rocsparse_spmat_descr A_descr_ = nullptr;
      rocsparse_spmat_descr B_descr_ = nullptr;
      rocsparse_spmat_descr D_descr_ = nullptr;

      long E_num_rows_ = 0;
      long E_num_cols_ = 0;
      long E_nnz_      = 0;

      index_type* E_row_ptr_ = nullptr;
      index_type* E_col_ind_ = nullptr;
      real_type*  E_val_     = nullptr;

      rocsparse_spmat_descr E_descr_ = nullptr;

      matrix::Csr** E_ptr_ = nullptr;

      size_t buffer_size_ = 0;
      void*  buffer_      = nullptr;

      rocsparse_spmat_descr convertToRocsparseType(matrix::Csr* A);
    };
  } // namespace hykkt
} // namespace ReSolve
