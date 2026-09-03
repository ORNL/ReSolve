#include "PreconditionerIChol0Cuda.hpp"

#include <resolve/utilities/logger/Logger.hpp>

#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>

namespace ReSolve
{
  using out = io::Logger;

  PreconditionerIChol0Cuda::PreconditionerIChol0Cuda(LinAlgWorkspaceCUDA* workspace)
  {
    workspace_ = workspace;
  }

  PreconditionerIChol0Cuda::~PreconditionerIChol0Cuda()
  {
    freeData();
    cusparseDestroyMatDescr(L_descr_);
  }

  void PreconditionerIChol0Cuda::freeData()
  {
    if (mat_L_)
    {
      cusparseDestroySpMat(mat_L_);
      mat_L_ = nullptr;
    }
    if (info_)
    {
      cusparseDestroyCsric02Info(info_);
      info_ = nullptr;
    }

    // SpSV (k = 1)
    if (L_descr_spsv_)
    {
      cusparseSpSV_destroyDescr(L_descr_spsv_);
      L_descr_spsv_ = nullptr;
    }
    if (L_tr_descr_spsv_)
    {
      cusparseSpSV_destroyDescr(L_tr_descr_spsv_);
      L_tr_descr_spsv_ = nullptr;
    }
    if (vec_b_)
    {
      cusparseDestroyDnVec(vec_b_);
      vec_b_ = nullptr;
    }
    if (vec_x_)
    {
      cusparseDestroyDnVec(vec_x_);
      vec_x_ = nullptr;
    }
    if (L_buffer_spsv_)
    {
      mem_.deleteOnDevice(L_buffer_spsv_);
      L_buffer_spsv_ = nullptr;
    }
    if (L_tr_buffer_spsv_)
    {
      mem_.deleteOnDevice(L_tr_buffer_spsv_);
      L_tr_buffer_spsv_ = nullptr;
    }
    L_buffer_size_spsv_ = 0;
    L_tr_buffer_size_spsv_ = 0;
    
    // SpSM (k > 1)
    if (L_descr_spsm_)
    {
      cusparseSpSM_destroyDescr(L_descr_spsm_);
      L_descr_spsm_ = nullptr;
    }
    if (L_tr_descr_spsm_)
    {
      cusparseSpSM_destroyDescr(L_tr_descr_spsm_);
      L_tr_descr_spsm_ = nullptr;
    }
    if (mat_B_)
    {
      cusparseDestroyDnMat(mat_B_);
      mat_B_ = nullptr;
    }
    if (mat_X_)
    {
      cusparseDestroyDnMat(mat_X_);
      mat_X_ = nullptr;
    }
    if (L_buffer_spsm_)
    {
      mem_.deleteOnDevice(L_buffer_spsm_);
      L_buffer_spsm_ = nullptr;
    }
    if (L_tr_buffer_spsm_)
    {
      mem_.deleteOnDevice(L_tr_buffer_spsm_);
      L_tr_buffer_spsm_ = nullptr;
    }

    L_buffer_size_spsm_ = 0;
    L_tr_buffer_size_spsm_ = 0;
  }

  int PreconditionerIChol0Cuda::setup(matrix::Csr* L)
  {
    L_ = L;

    cusparseCreateMatDescr(&L_descr_);
    cusparseSetMatIndexBase(L_descr_, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(L_descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(L_descr_, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(L_descr_, CUSPARSE_DIAG_TYPE_NON_UNIT);

    cusparseCreateCsric02Info(&info_);

    int status = 0;
    index_type n = L_->getNumRows();
    index_type nnz = L_->getNnz();

    int buffer_size = 0;
    status += cusparseDcsric02_bufferSize(workspace_->getCusparseHandle(),
                                          n,
                                          nnz,
                                          L_descr_,
                                          L_->getValues(memory::DEVICE),
                                          L_->getRowData(memory::DEVICE),
                                          L_->getColData(memory::DEVICE),
                                          info_,
                                          &buffer_size);

    void* buffer = nullptr;
    status += mem_.allocateBufferOnDevice(&buffer, static_cast<size_t>(buffer_size));

    status += cusparseDcsric02_analysis(workspace_->getCusparseHandle(),
                                         n,
                                         nnz,
                                         L_descr_,
                                         L_->getValues(memory::DEVICE),
                                         L_->getRowData(memory::DEVICE),
                                         L_->getColData(memory::DEVICE),
                                         info_,
                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                         buffer);

    int position = -1;
    status += cusparseXcsric02_zeroPivot(workspace_->getCusparseHandle(), info_, &position);
    if (position != -1)
    {
      out::error() << "Matrix is not SPD!";
    }
    status += (position != -1);

    status += cusparseDcsric02(workspace_->getCusparseHandle(),
                               n,
                               nnz,
                               L_descr_,
                               L_->getValues(memory::DEVICE),
                               L_->getRowData(memory::DEVICE),
                               L_->getColData(memory::DEVICE),
                               info_,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                               buffer);

    status += cusparseXcsric02_zeroPivot(workspace_->getCusparseHandle(), info_, &position);

    L_->setUpdated(memory::DEVICE);

    mem_.deleteOnDevice(buffer);

    status += analysis();

    return status;
  }

  int PreconditionerIChol0Cuda::analysis()
  {
    int status = 0;
    index_type n = L_->getNumRows();
    index_type nnz = L_->getNnz();
    index_type k = k_;

    // Create descriptor for L
    status += cusparseCreateCsr(&mat_L_,
                                n,
                                n,
                                nnz,
                                L_->getRowData(memory::DEVICE),
                                L_->getColData(memory::DEVICE),
                                L_->getValues(memory::DEVICE),
                                CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_64F);

    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    status += cusparseSpMatSetAttribute(mat_L_, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
    status += cusparseSpMatSetAttribute(mat_L_, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit, sizeof(diag_non_unit));

    static constexpr real_type alpha = 1.0;

    if (k_ == 1)
    {
      // SpSV
      status += cusparseCreateDnVec(&vec_b_, n, L_->getValues(memory::DEVICE), CUDA_R_64F);
      status += cusparseCreateDnVec(&vec_x_, n, L_->getValues(memory::DEVICE), CUDA_R_64F);

      status += cusparseSpSV_createDescr(&L_descr_spsv_);
      status += cusparseSpSV_createDescr(&L_tr_descr_spsv_);

      status += cusparseSpSV_bufferSize(workspace_->getCusparseHandle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        mat_L_,
                                        vec_b_,
                                        vec_x_,
                                        CUDA_R_64F,
                                        CUSPARSE_SPSV_ALG_DEFAULT,
                                        L_descr_spsv_,
                                        &L_buffer_size_spsv_);
      status += cusparseSpSV_bufferSize(workspace_->getCusparseHandle(),
                                        CUSPARSE_OPERATION_TRANSPOSE,
                                        &alpha,
                                        mat_L_,
                                        vec_x_,
                                        vec_x_,
                                        CUDA_R_64F,
                                        CUSPARSE_SPSV_ALG_DEFAULT,
                                        L_tr_descr_spsv_,
                                        &L_tr_buffer_size_spsv_);

      status += mem_.allocateBufferOnDevice(&L_buffer_spsv_, L_buffer_size_spsv_);
      status += mem_.allocateBufferOnDevice(&L_tr_buffer_spsv_, L_tr_buffer_size_spsv_);

      status += cusparseSpSV_analysis(workspace_->getCusparseHandle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      mat_L_,
                                      vec_b_,
                                      vec_x_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      L_descr_spsv_,
                                      L_buffer_spsv_);
      status += cusparseSpSV_analysis(workspace_->getCusparseHandle(),
                                      CUSPARSE_OPERATION_TRANSPOSE,
                                      &alpha,
                                      mat_L_,
                                      vec_x_,
                                      vec_x_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      L_tr_descr_spsv_,
                                      L_tr_buffer_spsv_);
    }
    else
    {
      // SpSM
      status += cusparseCreateDnMat(&mat_B_, n, k, n, L_->getValues(memory::DEVICE), CUDA_R_64F, CUSPARSE_ORDER_COL);
      status += cusparseCreateDnMat(&mat_X_, n, k, n, L_->getValues(memory::DEVICE), CUDA_R_64F, CUSPARSE_ORDER_COL);

      status += cusparseSpSM_createDescr(&L_descr_spsm_);
      status += cusparseSpSM_createDescr(&L_tr_descr_spsm_);

      status += cusparseSpSM_bufferSize(workspace_->getCusparseHandle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        mat_L_,
                                        mat_B_,
                                        mat_X_,
                                        CUDA_R_64F,
                                        CUSPARSE_SPSM_ALG_DEFAULT,
                                        L_descr_spsm_,
                                        &L_buffer_size_spsm_);
      status += cusparseSpSM_bufferSize(workspace_->getCusparseHandle(),
                                        CUSPARSE_OPERATION_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        mat_L_,
                                        mat_X_,
                                        mat_X_,
                                        CUDA_R_64F,
                                        CUSPARSE_SPSM_ALG_DEFAULT,
                                        L_tr_descr_spsm_,
                                        &L_tr_buffer_size_spsm_);

      status += cudaMalloc(&L_buffer_spsm_, L_buffer_size_spsm_);
      status += cudaMalloc(&L_tr_buffer_spsm_, L_tr_buffer_size_spsm_);

      status += cusparseSpSM_analysis(workspace_->getCusparseHandle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      mat_L_,
                                      mat_B_,
                                      mat_X_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSM_ALG_DEFAULT,
                                      L_descr_spsm_,
                                      L_buffer_spsm_);
      status += cusparseSpSM_analysis(workspace_->getCusparseHandle(),
                                      CUSPARSE_OPERATION_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      mat_L_,
                                      mat_X_,
                                      mat_X_,
                                      CUDA_R_64F,
                                      CUSPARSE_SPSM_ALG_DEFAULT,
                                      L_tr_descr_spsm_,
                                      L_tr_buffer_spsm_);
    }

    return status;
  }

  void PreconditionerIChol0Cuda::setNumRhs(index_type num_rhs)
  {
    if (k_ == num_rhs) return;
    k_ = num_rhs;
    if (L_)
    {
      freeData();
      analysis();
    }
  }

  int PreconditionerIChol0Cuda::apply(vector::Vector* rhs, vector::Vector* x)
  {
    static constexpr real_type alpha = 1.0;

    int status = 0;

    if (k_ == 1)
    {
      cusparseDnVecSetValues(vec_b_, rhs->getData(memory::DEVICE));
      cusparseDnVecSetValues(vec_x_, x->getData(memory::DEVICE));

      status += cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   mat_L_,
                                   vec_b_,
                                   vec_x_,
                                   CUDA_R_64F,
                                   CUSPARSE_SPSV_ALG_DEFAULT,
                                   L_descr_spsv_);
      status += cusparseSpSV_solve(workspace_->getCusparseHandle(),
                                   CUSPARSE_OPERATION_TRANSPOSE,
                                   &alpha,
                                   mat_L_,
                                   vec_x_,
                                   vec_x_,
                                   CUDA_R_64F,
                                   CUSPARSE_SPSV_ALG_DEFAULT,
                                   L_tr_descr_spsv_);
    }
    else
    {
      cusparseDnMatSetValues(mat_B_, rhs->getData(memory::DEVICE));
      cusparseDnMatSetValues(mat_X_, x->getData(memory::DEVICE));

      status += cusparseSpSM_solve(workspace_->getCusparseHandle(),
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   mat_L_,
                                   mat_B_,
                                   mat_X_,
                                   CUDA_R_64F,
                                   CUSPARSE_SPSM_ALG_DEFAULT,
                                   L_descr_spsm_);
      status += cusparseSpSM_solve(workspace_->getCusparseHandle(),
                                   CUSPARSE_OPERATION_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha,
                                   mat_L_,
                                   mat_X_,
                                   mat_X_,
                                   CUDA_R_64F,
                                   CUSPARSE_SPSM_ALG_DEFAULT,
                                   L_tr_descr_spsm_);
    }

    x->setDataUpdated(memory::DEVICE);

    return status;
  }
} // namespace ReSolve