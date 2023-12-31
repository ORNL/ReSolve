#include <algorithm>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>
#include <resolve/hip/hipKernels.h>
#include "MatrixHandlerHip.hpp"

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  MatrixHandlerHip::~MatrixHandlerHip()
  {
  }

  MatrixHandlerHip::MatrixHandlerHip(LinAlgWorkspaceHIP* new_workspace)
  {
    workspace_ = new_workspace;
  }

  void MatrixHandlerHip::setValuesChanged(bool values_changed)
  {
    values_changed_ = values_changed;
  }


  int MatrixHandlerHip::matvec(matrix::Sparse* Ageneric, 
                               vector_type* vec_x, 
                               vector_type* vec_result, 
                               const real_type* alpha, 
                               const real_type* beta,
                               std::string matrixFormat) 
  {
    using namespace constants;
    int error_sum = 0;
    if (matrixFormat == "csr") {
      matrix::Csr* A = dynamic_cast<matrix::Csr*>(Ageneric);
      //result = alpha *A*x + beta * result
      rocsparse_status status;

      rocsparse_handle handle_rocsparse = workspace_->getRocsparseHandle();
      
      rocsparse_mat_info infoA = workspace_->getSpmvMatrixInfo();
      rocsparse_mat_descr descrA =  workspace_->getSpmvMatrixDescriptor();
      
      if (!workspace_->matvecSetup()) {
        //setup first, allocate, etc.
        rocsparse_create_mat_descr(&(descrA));
        rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero);
        rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general);

        rocsparse_create_mat_info(&infoA);
        
        status = rocsparse_dcsrmv_analysis(handle_rocsparse,
                                           rocsparse_operation_none,
                                           A->getNumRows(),
                                           A->getNumColumns(),
                                           A->getNnzExpanded(), 
                                           descrA,
                                           A->getValues( memory::DEVICE), 
                                           A->getRowData(memory::DEVICE),
                                           A->getColData(memory::DEVICE), // cuda is used as "device"
                                           infoA);
        error_sum += status;
        mem_.deviceSynchronize();

        workspace_->setSpmvMatrixDescriptor(descrA);
        workspace_->setSpmvMatrixInfo(infoA);
        workspace_->matvecSetupDone();
      } 
      
      status = rocsparse_dcsrmv(handle_rocsparse,
                                rocsparse_operation_none,
                                A->getNumRows(),
                                A->getNumColumns(),
                                A->getNnzExpanded(),
                                alpha, 
                                descrA,
                                A->getValues( memory::DEVICE), 
                                A->getRowData(memory::DEVICE),
                                A->getColData(memory::DEVICE),
                                infoA,
                                vec_x->getData(memory::DEVICE),
                                beta,
                                vec_result->getData(memory::DEVICE));

      error_sum += status;
      mem_.deviceSynchronize();
      if (status)
        out::error() << "Matvec status: "   << status                    << ". "
                     << "Last error code: " << mem_.getLastDeviceError() << ".\n";
      vec_result->setDataUpdated(memory::DEVICE);

      return error_sum;
    } else {
      out::error() << "MatVec not implemented (yet) for " 
                   << matrixFormat << " matrix format." << std::endl;
      return 1;
    }
  }

  int MatrixHandlerHip::matrixInfNorm(matrix::Sparse* A, real_type* norm)
  {
    // we assume A is in CSR format
    real_type* d_r = workspace_->getDr();
    index_type d_r_size = workspace_->getDrSize();
    
    if (d_r_size != A->getNumRows()) {
      if (d_r_size != 0) {
        mem_.deleteOnDevice(d_r);
      }
      mem_.allocateArrayOnDevice(&d_r, A->getNumRows());
      workspace_->setDrSize(A->getNumRows());
      workspace_->setDr(d_r);
    }
    
    if (workspace_->getNormBufferState() == false) { // not allocated  
      real_type* buffer;
      mem_.allocateArrayOnDevice(&buffer, 1024);
      workspace_->setNormBuffer(buffer);
      workspace_->setNormBufferState(true);
    }

    mem_.deviceSynchronize();
    matrix_row_sums(A->getNumRows(),
                    A->getNnzExpanded(),
                    A->getRowData(memory::DEVICE),
                    A->getValues(memory::DEVICE),
                    d_r);
    mem_.deviceSynchronize();

    vector_inf_norm(A->getNumRows(),  
                    d_r, 
                    workspace_->getNormBuffer(),
                    norm);
    return 0;
  }

  int MatrixHandlerHip::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    index_type error_sum = 0;

    rocsparse_status status;
    
    A_csr->allocateMatrixData(memory::DEVICE);
    index_type n = A_csc->getNumRows();
    index_type m = A_csc->getNumRows();
    index_type nnz = A_csc->getNnz();
    size_t bufferSize;
    void* d_work;

    status = rocsparse_csr2csc_buffer_size(workspace_->getRocsparseHandle(),
                                           n,
                                           m,
                                           nnz,
                                           A_csc->getColData(memory::DEVICE), 
                                           A_csc->getRowData(memory::DEVICE), 
                                           rocsparse_action_numeric,
                                           &bufferSize);

    error_sum += status;
    mem_.allocateBufferOnDevice(&d_work, bufferSize);
    
    status = rocsparse_dcsr2csc(workspace_->getRocsparseHandle(),
                                n,
                                m,
                                nnz,
                                A_csc->getValues( memory::DEVICE), 
                                A_csc->getColData(memory::DEVICE), 
                                A_csc->getRowData(memory::DEVICE), 
                                A_csr->getValues( memory::DEVICE), 
                                A_csr->getRowData(memory::DEVICE),
                                A_csr->getColData(memory::DEVICE), 
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                d_work);
    error_sum += status;
    return error_sum;
    mem_.deleteOnDevice(d_work);
  }

} // namespace ReSolve
