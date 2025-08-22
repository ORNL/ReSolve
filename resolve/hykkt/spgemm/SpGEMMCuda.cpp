#include "SpGEMMCuda.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;
  using out       = ReSolve::io::Logger;

  namespace hykkt
  {
    SpGEMMCuda::SpGEMMCuda(real_type alpha, real_type beta)
      : alpha_(alpha), beta_(beta)
    {
      cusparseCreate(&handle_);
      cusparseSpGEMM_createDescr(&spgemm_desc_);
      cusparseCreateMatDescr(&D_descr_);
      cusparseSetMatType(D_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(D_descr_, CUSPARSE_INDEX_BASE_ZERO);
    }

    SpGEMMCuda::~SpGEMMCuda()
    {
      cusparseSpGEMM_destroyDescr(spgemm_desc_);
      cusparseDestroy(handle_);
    }

    void SpGEMMCuda::addProductMatrices(matrix::Csr* A, matrix::Csr* B)
    {
      A_descr_ = convertToCusparseType(A);
      B_descr_ = convertToCusparseType(B);

      n_ = A->getNumRows();

      mem_.allocateArrayOnDevice(&C_row_ptr_, (index_type) n_ + 1);

      cusparseCreateCsr(&C_descr_,
                        n_,
                        B->getNumColumns(),
                        0, // nnz will be determined later
                        C_row_ptr_,
                        nullptr,
                        nullptr,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F);
    }

    void SpGEMMCuda::addSumMatrix(matrix::Csr* D)
    {
      D_ = D;
    }

    void SpGEMMCuda::addResultMatrix(matrix::Csr** E_ptr)
    {
      E_ptr_ = E_ptr;
    }

    void SpGEMMCuda::compute()
    {
      double beta_product = 0.0;
      double alpha_sum    = 1.0;

      if (!(*E_ptr_))
      {
        size_t temp_buffer_size_1 = 0;
        size_t temp_buffer_size_2 = 0;
        size_t temp_buffer_size_3 = 0;

        void* temp_buffer_1 = nullptr;
        void* temp_buffer_2 = nullptr;
        void* temp_buffer_3 = nullptr;

        cusparseSpGEMMreuse_workEstimation(handle_,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           A_descr_,
                                           B_descr_,
                                           C_descr_,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemm_desc_,
                                           &temp_buffer_size_1,
                                           nullptr);

        mem_.allocateBufferOnDevice(&temp_buffer_1, temp_buffer_size_1);

        cusparseSpGEMMreuse_workEstimation(handle_,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           A_descr_,
                                           B_descr_,
                                           C_descr_,
                                           CUSPARSE_SPGEMM_DEFAULT,
                                           spgemm_desc_,
                                           &temp_buffer_size_1,
                                           temp_buffer_1);

        cusparseSpGEMMreuse_nnz(handle_,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                A_descr_,
                                B_descr_,
                                C_descr_,
                                CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc_,
                                &temp_buffer_size_2,
                                nullptr,
                                &temp_buffer_size_3,
                                nullptr,
                                &buffer_size_4_,
                                nullptr);

        mem_.allocateBufferOnDevice(&temp_buffer_2, temp_buffer_size_2);
        mem_.allocateBufferOnDevice(&temp_buffer_3, temp_buffer_size_3);
        mem_.allocateBufferOnDevice(&buffer_4_, buffer_size_4_);

        cusparseSpGEMMreuse_nnz(handle_,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                A_descr_,
                                B_descr_,
                                C_descr_,
                                CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc_,
                                &temp_buffer_size_2,
                                temp_buffer_2,
                                &temp_buffer_size_3,
                                temp_buffer_3,
                                &buffer_size_4_,
                                buffer_4_);

        mem_.deleteOnDevice(temp_buffer_1);
        mem_.deleteOnDevice(temp_buffer_2);

        int64_t C_num_cols = 0;
        int64_t C_nnz_     = 0;
        cusparseSpMatGetSize(C_descr_, &n_, &C_num_cols, &C_nnz_);

        mem_.allocateArrayOnDevice(&C_col_ind_, (index_type) C_nnz_);
        mem_.allocateArrayOnDevice(&C_val_, (index_type) C_nnz_);

        cusparseCsrSetPointers(C_descr_,
                               C_row_ptr_,
                               C_col_ind_,
                               C_val_);

        cusparseSpGEMMreuse_copy(handle_,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 A_descr_,
                                 B_descr_,
                                 C_descr_,
                                 CUSPARSE_SPGEMM_DEFAULT,
                                 spgemm_desc_,
                                 &buffer_size_5_,
                                 nullptr);

        mem_.allocateBufferOnDevice(&buffer_5_, buffer_size_5_);

        cusparseSpGEMMreuse_copy(handle_,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 A_descr_,
                                 B_descr_,
                                 C_descr_,
                                 CUSPARSE_SPGEMM_DEFAULT,
                                 spgemm_desc_,
                                 &buffer_size_5_,
                                 buffer_5_);

        mem_.deleteOnDevice(temp_buffer_3);
      }

      cusparseSpGEMMreuse_compute(handle_,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha_,
                                  A_descr_,
                                  B_descr_,
                                  &beta_product,
                                  C_descr_,
                                  CUDA_R_64F,
                                  CUSPARSE_SPGEMM_DEFAULT,
                                  spgemm_desc_);

      if (!(*E_ptr_))
      {
        // Begin set up for addition
        mem_.allocateArrayOnDevice(&E_row_ptr_, (index_type) n_ + 1);
        cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);

        // pass in D_descr_ as a dummy variable
        cusparseDcsrgeam2_bufferSizeExt(handle_,
                                        D_->getNumRows(),
                                        D_->getNumColumns(),
                                        &alpha_sum,
                                        D_descr_,
                                        (int) C_nnz_,
                                        C_val_,
                                        C_row_ptr_,
                                        C_col_ind_,
                                        &beta_,
                                        D_descr_,
                                        D_->getNnz(),
                                        D_->getValues(memory::DEVICE),
                                        D_->getRowData(memory::DEVICE),
                                        D_->getColData(memory::DEVICE),
                                        D_descr_,
                                        E_val_,
                                        E_row_ptr_,
                                        E_col_ind_,
                                        &buffer_add_size_);

        mem_.allocateBufferOnDevice(&buffer_add_, buffer_add_size_);

        cusparseXcsrgeam2Nnz(handle_,
                             D_->getNumRows(),
                             D_->getNumColumns(),
                             D_descr_,
                             (int) C_nnz_,
                             C_row_ptr_,
                             C_col_ind_,
                             D_descr_,
                             D_->getNnz(),
                             D_->getRowData(memory::DEVICE),
                             D_->getColData(memory::DEVICE),
                             D_descr_,
                             E_row_ptr_,
                             &E_nnz_,
                             buffer_add_);

        mem_.allocateArrayOnDevice(&E_col_ind_, (index_type) E_nnz_);
        mem_.allocateArrayOnDevice(&E_val_, (index_type) E_nnz_);

        (*E_ptr_) = new matrix::Csr((index_type) n_, D_->getNumColumns(), (index_type) E_nnz_);
        (*E_ptr_)->setDataPointers(E_row_ptr_, E_col_ind_, E_val_, memory::DEVICE);
      }

      cusparseDcsrgeam2(handle_,
                        D_->getNumRows(),
                        D_->getNumColumns(),
                        &alpha_sum,
                        D_descr_,
                        (int) C_nnz_,
                        C_val_,
                        C_row_ptr_,
                        C_col_ind_,
                        &beta_,
                        D_descr_,
                        D_->getNnz(),
                        D_->getValues(memory::DEVICE),
                        D_->getRowData(memory::DEVICE),
                        D_->getColData(memory::DEVICE),
                        D_descr_,
                        E_val_,
                        E_row_ptr_,
                        E_col_ind_,
                        buffer_add_);
    }

    cusparseSpMatDescr_t SpGEMMCuda::convertToCusparseType(matrix::Csr* A)
    {
      cusparseSpMatDescr_t descr;
      // TODO: this hardcodes the types but should be based on ReSolve types
      cusparseCreateCsr(&descr,
                        A->getNumRows(),
                        A->getNumColumns(),
                        A->getNnz(),
                        A->getRowData(memory::DEVICE),
                        A->getColData(memory::DEVICE),
                        A->getValues(memory::DEVICE),
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F);
      return descr;
    }
  } // namespace hykkt
} // namespace ReSolve
