#include "SpGEMMHip.hpp"

namespace ReSolve {
  using real_type = ReSolve::real_type;
  using out = ReSolve::io::Logger;

  namespace hykkt {
    SpGEMMHip::SpGEMMHip(real_type alpha, real_type beta): alpha_(alpha), beta_(beta)
    {
      rocsparse_create_handle(&handle_);
    }

    SpGEMMHip::~SpGEMMHip()
    {
      rocsparse_destroy_spmat_descr(A_descr_);
      rocsparse_destroy_spmat_descr(B_descr_);
      rocsparse_destroy_spmat_descr(D_descr_);
      rocsparse_destroy_spmat_descr(E_descr_);
      rocsparse_destroy_handle(handle_);

      mem_.deleteOnDevice(buffer_);
    }

    void SpGEMMHip::addProductMatrices(matrix::Csr* A, matrix::Csr* B)
    {
      A_descr_ = convertToRocsparseType(A);
      B_descr_ = convertToRocsparseType(B);

      E_num_rows_ = A->getNumRows();
      E_num_cols_ = B->getNumColumns();
    }

    void SpGEMMHip::addSumMatrix(matrix::Csr* D)
    {
      D_descr_ = convertToRocsparseType(D);
    }

    void SpGEMMHip::addResultMatrix(matrix::Csr** E_ptr)
    {
      // TODO: will this ever be called more than once?
      if (!E_ptr_)
      {
        mem_.allocateArrayOnDevice(&E_row_ptr_, (index_type) E_num_rows_+1);
        rocsparse_create_csr_descr(&E_descr_,
                E_num_rows_,
                E_num_cols_,
                E_nnz_,
                E_row_ptr_,
                nullptr,
                nullptr,
                rocsparse_indextype_i32,
                rocsparse_indextype_i32,
                rocsparse_index_base_zero,
                rocsparse_datatype_f64_r);
      }
      E_ptr_ = E_ptr;
    }

    void SpGEMMHip::compute()
    {
      rocsparse_status status;
      if (!buffer_) // first computation
      {
        // Determine buffer size and allocate
        status = rocsparse_spgemm(handle_,
                  rocsparse_operation_none,
                  rocsparse_operation_none,
                  &alpha_,
                  A_descr_,
                  B_descr_,
                  &beta_,
                  D_descr_,
                  E_descr_,
                  rocsparse_datatype_f64_r,
                  rocsparse_spgemm_alg_default,
                  rocsparse_spgemm_stage_buffer_size,
                  &buffer_size_,
                  nullptr);
        if (status != rocsparse_status_success)
        {
          out::error() << "Failed to determine buffer size. Status: " << status << "\n";
        }
        mem_.allocateBufferOnDevice(&buffer_, buffer_size_);

        // Determine number of nonzeros in result
        status = rocsparse_spgemm(handle_,
                 rocsparse_operation_none,
                 rocsparse_operation_none,
                 &alpha_,
                 A_descr_,
                 B_descr_,
                 &beta_,
                 D_descr_,
                 E_descr_,
                 rocsparse_datatype_f64_r,
                 rocsparse_spgemm_alg_default,
                 rocsparse_spgemm_stage_nnz,
                 &buffer_size_,
                 buffer_);
        if (status != rocsparse_status_success)
        {
          out::error() << "Failed to determine number of nonzeros. Status: " << status << "\n";
        }

        rocsparse_spmat_get_size(E_descr_, &E_num_rows_, &E_num_cols_, &E_nnz_);
      
        mem_.allocateArrayOnDevice(&E_col_ind_, (index_type) E_nnz_);
        mem_.allocateArrayOnDevice(&E_val_, (index_type) E_nnz_);

        rocsparse_csr_set_pointers(E_descr_, E_row_ptr_, E_col_ind_, E_val_);

        *E_ptr_ = new matrix::Csr((index_type) E_num_rows_, (index_type) E_num_cols_, (index_type) E_nnz_);
        (*E_ptr_)->setDataPointers(E_row_ptr_, E_col_ind_, E_val_, memory::DEVICE);
      }
      
      // SpGEMM computation
      status = rocsparse_spgemm(handle_,
                      rocsparse_operation_none,
                      rocsparse_operation_none,
                      &alpha_,
                      A_descr_,
                      B_descr_,
                      &beta_,
                      D_descr_,
                      E_descr_,
                      rocsparse_datatype_f64_r,
                      rocsparse_spgemm_alg_default,
                      rocsparse_spgemm_stage_compute,
                      &buffer_size_,
                      buffer_);
      if (status != rocsparse_status_success)
      {
        out::error() << "Failed to compute SpGEMM. Status: " << status << "\n";
      }
    }

    rocsparse_spmat_descr SpGEMMHip::convertToRocsparseType(matrix::Csr* A)
    {
      rocsparse_spmat_descr descr;
      // TODO: this hardcodes the types but should be based on ReSolve types
      rocsparse_create_csr_descr(&descr,
        A->getNumRows(),
        A->getNumColumns(),
        A->getNnz(),
        A->getRowData(memory::DEVICE),
        A->getColData(memory::DEVICE),
        A->getValues(memory::DEVICE),
        rocsparse_indextype_i32,
        rocsparse_indextype_i32,
        rocsparse_index_base_zero,
        rocsparse_datatype_f64_r);
      return descr;
    }
  }
}