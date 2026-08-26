// ...
// This uses rocSPARSE srsv with multiple streams instead of srsm because srsm is very slow. Maybe it will be different on different hardware.

#include "PreconditionerIChol0Hip.hpp"

#include <resolve/utilities/logger/Logger.hpp>

#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

namespace ReSolve
{
  using out = io::Logger;

  PreconditionerIChol0Hip::PreconditionerIChol0Hip(LinAlgWorkspaceHIP* workspace)
  {
    workspace_ = workspace;
  }

  PreconditionerIChol0Hip::~PreconditionerIChol0Hip()
  {
    freeData();
    if (L_descr_)
    {
      rocsparse_destroy_mat_descr(L_descr_);
    }
  }

  void PreconditionerIChol0Hip::freeData()
  {
    for (hipStream_t stream : streams_)
    {
      hipStreamDestroy(stream);
    }
    streams_.clear();

    if (start_event_)
    {
      hipEventDestroy(start_event_);
      start_event_ = nullptr;
    }

    for (hipEvent_t event : end_events_)
    {
      hipEventDestroy(event);
    }
    end_events_.clear();

    for (rocsparse_mat_info info : L_info_)
    {
      rocsparse_destroy_mat_info(info);
    }
    L_info_.clear();

    for (rocsparse_mat_info info : L_tr_info_)
    {
      rocsparse_destroy_mat_info(info);
    }
    L_tr_info_.clear();

    if (L_buffer_)
    {
      hipFree(L_buffer_);
      L_buffer_ = nullptr;
    }

    if (L_tr_buffer_)
    {
      hipFree(L_tr_buffer_);
      L_tr_buffer_ = nullptr;
    }
  }

  int PreconditionerIChol0Hip::setup(matrix::Csr* L)
  {
    L_ = L;

    rocsparse_create_mat_descr(&L_descr_);
    rocsparse_set_mat_index_base(L_descr_, rocsparse_index_base_zero);
    rocsparse_set_mat_type(L_descr_, rocsparse_matrix_type_general);
    rocsparse_set_mat_fill_mode(L_descr_, rocsparse_fill_mode_lower);
    rocsparse_set_mat_diag_type(L_descr_, rocsparse_diag_type_non_unit);

    rocsparse_mat_info L_info_setup;
    rocsparse_create_mat_info(&L_info_setup);

    int status = 0;

    size_t buffer_size = 0;
    status += rocsparse_dcsric0_buffer_size(workspace_->getRocsparseHandle(),
                                            L_->getNumRows(),
                                            L_->getNnz(),
                                            L_descr_,
                                            L_->getValues(memory::DEVICE),
                                            L_->getRowData(memory::DEVICE),
                                            L_->getColData(memory::DEVICE),
                                            L_info_setup,
                                            &buffer_size);

    void* buffer;
    status += hipMalloc(&buffer, buffer_size);
    status += rocsparse_dcsric0_analysis(workspace_->getRocsparseHandle(),
                                         L_->getNumRows(),
                                         L_->getNnz(),
                                         L_descr_,
                                         L_->getValues(memory::DEVICE),
                                         L_->getRowData(memory::DEVICE),
                                         L_->getColData(memory::DEVICE),
                                         L_info_setup,
                                         rocsparse_analysis_policy_force,
                                         rocsparse_solve_policy_auto,
                                         buffer);

    int position = -1;
    status += rocsparse_csric0_zero_pivot(workspace_->getRocsparseHandle(), L_info_setup, &position);
    if (position != -1)
    {
      out::error() << "Matrix is not SPD!";
    }
    status += (position != -1);

    status += rocsparse_dcsric0(workspace_->getRocsparseHandle(),
                             L_->getNumRows(),
                             L_->getNnz(),
                             L_descr_,
                             L_->getValues(memory::DEVICE),
                             L_->getRowData(memory::DEVICE),
                             L_->getColData(memory::DEVICE),
                             L_info_setup,
                             rocsparse_solve_policy_auto,
                             buffer);

    status += rocsparse_csric0_zero_pivot(workspace_->getRocsparseHandle(), L_info_setup, &position);
    if (position != -1)
    {
      out::warning() << "Numerical zero pivot found. Increase numeric boost to compensate.";
    }
    status += (position != -1);

    L_->setUpdated(memory::DEVICE);

    hipFree(buffer);
    rocsparse_destroy_mat_info(L_info_setup);

    status += analysis();

    return status;
  }

  int PreconditionerIChol0Hip::analysis()
  {
    index_type n = L_->getNumRows();
    int status = 0;

    L_info_.resize(k_);
    L_tr_info_.resize(k_);

    for (index_type i = 0; i < k_; ++i)
    {
      rocsparse_create_mat_info(&L_info_[i]);
      rocsparse_create_mat_info(&L_tr_info_[i]);
    }



























    
    status += rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                            rocsparse_operation_none,
                                            n,
                                            L_->getNnz(),
                                            L_descr_,
                                            L_->getValues(memory::DEVICE),
                                            L_->getRowData(memory::DEVICE),
                                            L_->getColData(memory::DEVICE),
                                            L_info_[0],
                                            &L_buffer_size_);

    status += rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(),
                                            rocsparse_operation_transpose,
                                            n,
                                            L_->getNnz(),
                                            L_descr_,
                                            L_->getValues(memory::DEVICE),
                                            L_->getRowData(memory::DEVICE),
                                            L_->getColData(memory::DEVICE),
                                            L_tr_info_[0],
                                            &L_tr_buffer_size_);

    status += hipMalloc(&L_buffer_, L_buffer_size_ * k_);
    status += hipMalloc(&L_tr_buffer_, L_tr_buffer_size_ * k_);

    for (index_type i = 0; i < k_; i++)
    {
      void* L_buf_i = static_cast<char*>(L_buffer_) + i * L_buffer_size_;
      void* L_tr_buf_i = static_cast<char*>(L_tr_buffer_) + i * L_tr_buffer_size_;

      status += rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                           rocsparse_operation_none,
                                           n,
                                           L_->getNnz(),
                                           L_descr_,
                                           L_->getValues(memory::DEVICE),
                                           L_->getRowData(memory::DEVICE),
                                           L_->getColData(memory::DEVICE),
                                           L_info_[i],
                                           rocsparse_analysis_policy_force,
                                           rocsparse_solve_policy_auto,
                                           L_buf_i);

      status += rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(),
                                           rocsparse_operation_transpose,
                                           n,
                                           L_->getNnz(),
                                           L_descr_,
                                           L_->getValues(memory::DEVICE),
                                           L_->getRowData(memory::DEVICE),
                                           L_->getColData(memory::DEVICE),
                                           L_tr_info_[i],
                                           rocsparse_analysis_policy_force,
                                           rocsparse_solve_policy_auto,
                                           L_tr_buf_i);
    }

    if (k_ > 1)
    {
      streams_.resize(k_);
      end_events_.resize(k_);
      status += hipEventCreate(&start_event_);

      for (index_type i = 0; i < k_; ++i)
      {
        status += hipStreamCreate(&streams_[i]);
        status += hipEventCreate(&end_events_[i]);
      }
    }

    return status;
  }

  void PreconditionerIChol0Hip::setNumRhs(index_type num_rhs)
  {
    if (k_ == num_rhs) return;
    k_ = num_rhs;
    if (L_)
    {
      freeData();
      analysis();
    }
  }

  int PreconditionerIChol0Hip::apply(vector::Vector* rhs, vector::Vector* x)
  {
    static constexpr real_type alpha = 1.0;
    index_type n = L_->getNumRows();

    const real_type* rhs_data = rhs->getData(memory::DEVICE);
    real_type* x_ptr = x->getData(memory::DEVICE);

    int status = 0;
    if (k_ == 1)
    {
      status += rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                      rocsparse_operation_none,
                                      n,
                                      L_->getNnz(),
                                      &alpha,
                                      L_descr_,
                                      L_->getValues(memory::DEVICE),
                                      L_->getRowData(memory::DEVICE),
                                      L_->getColData(memory::DEVICE),
                                      L_info_[0],
                                      rhs_data,
                                      x_ptr,
                                      rocsparse_solve_policy_auto,
                                      L_buffer_);

      status += rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                      rocsparse_operation_transpose,
                                      n,
                                      L_->getNnz(),
                                      &alpha,
                                      L_descr_,
                                      L_->getValues(memory::DEVICE),
                                      L_->getRowData(memory::DEVICE),
                                      L_->getColData(memory::DEVICE),
                                      L_tr_info_[0],
                                      x_ptr,
                                      x_ptr,
                                      rocsparse_solve_policy_auto,
                                      L_tr_buffer_);
    }
    else
    {
      hipStream_t main_stream;
      rocsparse_get_stream(workspace_->getRocsparseHandle(), &main_stream);

      hipEventRecord(start_event_, main_stream);

      for (index_type i = 0; i < k_; i++)
      {
        hipStreamWaitEvent(streams_[i], start_event_, 0);
        rocsparse_set_stream(workspace_->getRocsparseHandle(), streams_[i]);

        const real_type* rhs_i = rhs_data + i * n;
        real_type* x_i = x_ptr + i * n;

        void* L_buf_i = static_cast<char*>(L_buffer_) + i * L_buffer_size_;
        void* L_tr_buf_i = static_cast<char*>(L_tr_buffer_) + i * L_tr_buffer_size_;

        status += rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                        rocsparse_operation_none,
                                        n,
                                        L_->getNnz(),
                                        &alpha,
                                        L_descr_,
                                        L_->getValues(memory::DEVICE),
                                        L_->getRowData(memory::DEVICE),
                                        L_->getColData(memory::DEVICE),
                                        L_info_[i],
                                        rhs_i,
                                        x_i,
                                        rocsparse_solve_policy_auto,
                                        L_buf_i);

        status += rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(),
                                        rocsparse_operation_transpose,
                                        n,
                                        L_->getNnz(),
                                        &alpha,
                                        L_descr_,
                                        L_->getValues(memory::DEVICE),
                                        L_->getRowData(memory::DEVICE),
                                        L_->getColData(memory::DEVICE),
                                        L_tr_info_[i],
                                        x_i,
                                        x_i,
                                        rocsparse_solve_policy_auto,
                                        L_tr_buf_i);

        hipEventRecord(end_events_[i], streams_[i]);
      }

      // Sync main stream with other streams
      for (index_type i = 0; i < k_; ++i)
      {
        hipStreamWaitEvent(main_stream, end_events_[i], 0);
      }

      rocsparse_set_stream(workspace_->getRocsparseHandle(), main_stream);
    }

    x->setDataUpdated(memory::DEVICE);


    return status;
  }
} // namespace ReSolve