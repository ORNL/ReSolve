#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include "LinSolverDirectRocSolverRf.hpp"
#include <resolve/hip/hipKernels.h>
#include <resolve/Profiling.hpp>

namespace ReSolve 
{
  LinSolverDirectRocSolverRf::LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace)
  {
    workspace_ = workspace;
    infoM_ = nullptr;
    solve_mode_ = 1; //solve mode - fast mode is default
  }

  LinSolverDirectRocSolverRf::~LinSolverDirectRocSolverRf()
  {
    mem_.deleteOnDevice(d_P_);
    mem_.deleteOnDevice(d_Q_);

    mem_.deleteOnDevice(d_aux1_);
    mem_.deleteOnDevice(d_aux2_);

    delete L_csr_;
    delete U_csr_;
  }

  int LinSolverDirectRocSolverRf::setup(matrix::Sparse* A,
                                        matrix::Sparse* L,
                                        matrix::Sparse* U,
                                        index_type* P,
                                        index_type* Q,
                                        vector_type* rhs)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    //set matrix info
    rocsolver_create_rfinfo(&infoM_, workspace_->getRocblasHandle()); 
    //create combined factor

    addFactors(L, U);

    M_->setUpdated(ReSolve::memory::HOST);
    M_->copyData(ReSolve::memory::DEVICE);

    if (d_P_ == nullptr) {
      mem_.allocateArrayOnDevice(&d_P_, n); 
    }

    if (d_Q_ == nullptr) {
      mem_.allocateArrayOnDevice(&d_Q_, n);
    }
    mem_.copyArrayHostToDevice(d_P_, P, n);
    mem_.copyArrayHostToDevice(d_Q_, Q, n);

    mem_.deviceSynchronize();
    status_rocblas_ = rocsolver_dcsrrf_analysis(workspace_->getRocblasHandle(),
                                                n,
                                                1,
                                                A_->getNnzExpanded(),
                                                A_->getRowData(ReSolve::memory::DEVICE), //kRowPtr_,
                                                A_->getColData(ReSolve::memory::DEVICE), //jCol_, 
                                                A_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                M_->getNnzExpanded(),
                                                M_->getRowData(ReSolve::memory::DEVICE), 
                                                M_->getColData(ReSolve::memory::DEVICE), 
                                                M_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                d_P_,
                                                d_Q_,
                                                rhs->getData(ReSolve::memory::DEVICE), 
                                                n,
                                                infoM_);

    mem_.deviceSynchronize();
    error_sum += status_rocblas_;
    // tri solve setup
    if (solve_mode_ == 1) { // fast mode

      if (L_csr_ != nullptr) {
        delete L_csr_;
      }

      L_csr_ = new ReSolve::matrix::Csr(L->getNumRows(), L->getNumColumns(), L->getNnz());
      L_csr_->allocateMatrixData(ReSolve::memory::DEVICE); 

      if (U_csr_ != nullptr) {
        delete U_csr_;
      }

      U_csr_ = new ReSolve::matrix::Csr(U->getNumRows(), U->getNumColumns(), U->getNnz());
      U_csr_->allocateMatrixData(ReSolve::memory::DEVICE); 


      rocsparse_create_mat_descr(&(descr_L_));
      rocsparse_set_mat_fill_mode(descr_L_, rocsparse_fill_mode_lower);
      rocsparse_set_mat_index_base(descr_L_, rocsparse_index_base_zero);

      rocsparse_create_mat_descr(&(descr_U_));
      rocsparse_set_mat_index_base(descr_U_, rocsparse_index_base_zero);
      rocsparse_set_mat_fill_mode(descr_U_, rocsparse_fill_mode_upper);

      rocsparse_create_mat_info(&info_L_);
      rocsparse_create_mat_info(&info_U_);

      // local variables
      size_t L_buffer_size;  
      size_t U_buffer_size;  

      status_rocblas_ = rocsolver_dcsrrf_splitlu(workspace_->getRocblasHandle(),
                                                 n,
                                                 M_->getNnzExpanded(),
                                                 M_->getRowData(ReSolve::memory::DEVICE), 
                                                 M_->getColData(ReSolve::memory::DEVICE), 
                                                 M_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                 L_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                 L_csr_->getColData(ReSolve::memory::DEVICE), 
                                                 L_csr_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                 U_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                 U_csr_->getColData(ReSolve::memory::DEVICE), 
                                                 U_csr_->getValues(ReSolve::memory::DEVICE));

      error_sum += status_rocblas_;

      status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(), 
                                                       rocsparse_operation_none, 
                                                       n, 
                                                       L_csr_->getNnz(), 
                                                       descr_L_,
                                                       L_csr_->getValues(ReSolve::memory::DEVICE),
                                                       L_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                       L_csr_->getColData(ReSolve::memory::DEVICE), 
                                                       info_L_, 
                                                       &L_buffer_size);
      error_sum += status_rocsparse_;

      mem_.allocateBufferOnDevice(&L_buffer_, L_buffer_size);
      status_rocsparse_ = rocsparse_dcsrsv_buffer_size(workspace_->getRocsparseHandle(), 
                                                       rocsparse_operation_none, 
                                                       n, 
                                                       U_csr_->getNnz(), 
                                                       descr_U_,
                                                       U_csr_->getValues(ReSolve::memory::DEVICE),
                                                       U_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                       U_csr_->getColData(ReSolve::memory::DEVICE), 
                                                       info_U_,
                                                       &U_buffer_size);
      error_sum += status_rocsparse_;
      mem_.allocateBufferOnDevice(&U_buffer_, U_buffer_size);

      status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(), 
                                                    rocsparse_operation_none, 
                                                    n, 
                                                    L_csr_->getNnz(), 
                                                    descr_L_,
                                                    L_csr_->getValues(ReSolve::memory::DEVICE),
                                                    L_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                    L_csr_->getColData(ReSolve::memory::DEVICE), 
                                                    info_L_,   
                                                    rocsparse_analysis_policy_force,
                                                    rocsparse_solve_policy_auto,
                                                    L_buffer_);
      error_sum += status_rocsparse_;
      if (status_rocsparse_!=0)printf("status after analysis 1 %d \n", status_rocsparse_);
      status_rocsparse_ = rocsparse_dcsrsv_analysis(workspace_->getRocsparseHandle(), 
                                                    rocsparse_operation_none, 
                                                    n, 
                                                    U_csr_->getNnz(), 
                                                    descr_U_,
                                                    U_csr_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                    U_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                    U_csr_->getColData(ReSolve::memory::DEVICE), 
                                                    info_U_,
                                                    rocsparse_analysis_policy_force,
                                                    rocsparse_solve_policy_auto,
                                                    U_buffer_);
      error_sum += status_rocsparse_;
      if (status_rocsparse_!=0)printf("status after analysis 2 %d \n", status_rocsparse_);
      //allocate aux data
      if (d_aux1_ == nullptr) {
        mem_.allocateArrayOnDevice(&d_aux1_,n); 
      }
      if (d_aux2_ == nullptr) { 
        mem_.allocateArrayOnDevice(&d_aux2_,n); 
      }

    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  int LinSolverDirectRocSolverRf::refactorize()
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum = 0;
    mem_.deviceSynchronize();
    status_rocblas_ =  rocsolver_dcsrrf_refactlu(workspace_->getRocblasHandle(),
                                                 A_->getNumRows(),
                                                 A_->getNnzExpanded(),
                                                 A_->getRowData(ReSolve::memory::DEVICE), //kRowPtr_,
                                                 A_->getColData(ReSolve::memory::DEVICE), //jCol_, 
                                                 A_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                 M_->getNnzExpanded(),
                                                 M_->getRowData(ReSolve::memory::DEVICE), 
                                                 M_->getColData(ReSolve::memory::DEVICE), 
                                                 M_->getValues(ReSolve::memory::DEVICE), //OUTPUT, 
                                                 d_P_,
                                                 d_Q_,
                                                 infoM_);

    mem_.deviceSynchronize();
    error_sum += status_rocblas_;

    if (solve_mode_ == 1) {
      //split M, fill L and U with correct values
      status_rocblas_ = rocsolver_dcsrrf_splitlu(workspace_->getRocblasHandle(),
                                                 A_->getNumRows(),
                                                 M_->getNnzExpanded(),
                                                 M_->getRowData(ReSolve::memory::DEVICE), 
                                                 M_->getColData(ReSolve::memory::DEVICE), 
                                                 M_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                 L_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                 L_csr_->getColData(ReSolve::memory::DEVICE), 
                                                 L_csr_->getValues(ReSolve::memory::DEVICE), //vals_, 
                                                 U_csr_->getRowData(ReSolve::memory::DEVICE), 
                                                 U_csr_->getColData(ReSolve::memory::DEVICE), 
                                                 U_csr_->getValues(ReSolve::memory::DEVICE));

      mem_.deviceSynchronize();
      error_sum += status_rocblas_;

    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectRocSolverRf::solve(vector_type* rhs)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    int error_sum = 0;
    if (solve_mode_ == 0) {
      mem_.deviceSynchronize();
      status_rocblas_ =  rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                                A_->getNumRows(),
                                                1,
                                                M_->getNnz(),
                                                M_->getRowData(ReSolve::memory::DEVICE), 
                                                M_->getColData(ReSolve::memory::DEVICE), 
                                                M_->getValues(ReSolve::memory::DEVICE), 
                                                d_P_,
                                                d_Q_,
                                                rhs->getData(ReSolve::memory::DEVICE),
                                                A_->getNumRows(),
                                                infoM_);
      mem_.deviceSynchronize();
    } else {
      // not implemented yet
      permuteVectorP(A_->getNumRows(), d_P_, rhs->getData(ReSolve::memory::DEVICE), d_aux1_);
      mem_.deviceSynchronize();
      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(), 
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             L_csr_->getNnz(), 
                             &(constants::ONE), 
                             descr_L_,
                             L_csr_->getValues(ReSolve::memory::DEVICE),
                             L_csr_->getRowData(ReSolve::memory::DEVICE), 
                             L_csr_->getColData(ReSolve::memory::DEVICE), 
                             info_L_,
                             d_aux1_,
                             d_aux2_, //result
                             rocsparse_solve_policy_auto,
                             L_buffer_);
      error_sum += status_rocsparse_;

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(), 
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             U_csr_->getNnz(), 
                             &(constants::ONE), 
                             descr_U_,
                             U_csr_->getValues(ReSolve::memory::DEVICE),
                             U_csr_->getRowData(ReSolve::memory::DEVICE), 
                             U_csr_->getColData(ReSolve::memory::DEVICE), 
                             info_U_,
                             d_aux2_, //input
                             d_aux1_, //result
                             rocsparse_solve_policy_auto,
                             U_buffer_);
      error_sum += status_rocsparse_;

      permuteVectorQ(A_->getNumRows(), d_Q_,d_aux1_,rhs->getData(ReSolve::memory::DEVICE));
      mem_.deviceSynchronize();
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  int LinSolverDirectRocSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    RESOLVE_RANGE_PUSH(__FUNCTION__);
    x->update(rhs->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
    x->setDataUpdated(ReSolve::memory::DEVICE);
    int error_sum = 0;
    if (solve_mode_ == 0) {
      mem_.deviceSynchronize();
      status_rocblas_ =  rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                                A_->getNumRows(),
                                                1,
                                                M_->getNnz(),
                                                M_->getRowData(ReSolve::memory::DEVICE), 
                                                M_->getColData(ReSolve::memory::DEVICE),  
                                                M_->getValues(ReSolve::memory::DEVICE), 
                                                d_P_,
                                                d_Q_,
                                                x->getData(ReSolve::memory::DEVICE),
                                                A_->getNumRows(),
                                                infoM_);
      error_sum += status_rocblas_;
      mem_.deviceSynchronize();
    } else {
      // not implemented yet

      permuteVectorP(A_->getNumRows(), d_P_, rhs->getData(ReSolve::memory::DEVICE), d_aux1_);
      mem_.deviceSynchronize();

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(), 
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             L_csr_->getNnz(), 
                             &(constants::ONE), 
                             descr_L_,
                             L_csr_->getValues(ReSolve::memory::DEVICE), //vals_, 
                             L_csr_->getRowData(ReSolve::memory::DEVICE), 
                             L_csr_->getColData(ReSolve::memory::DEVICE), 
                             info_L_,
                             d_aux1_,
                             d_aux2_, //result
                             rocsparse_solve_policy_auto,
                             L_buffer_);
      error_sum += status_rocsparse_;

      rocsparse_dcsrsv_solve(workspace_->getRocsparseHandle(), 
                             rocsparse_operation_none,
                             A_->getNumRows(),
                             U_csr_->getNnz(), 
                             &(constants::ONE), 
                             descr_U_,
                             U_csr_->getValues(ReSolve::memory::DEVICE), //vals_, 
                             U_csr_->getRowData(ReSolve::memory::DEVICE), 
                             U_csr_->getColData(ReSolve::memory::DEVICE), 
                             info_U_,
                             d_aux2_, //input
                             d_aux1_,//result
                             rocsparse_solve_policy_auto,
                             U_buffer_);
      error_sum += status_rocsparse_;

      permuteVectorQ(A_->getNumRows(), d_Q_,d_aux1_,x->getData(ReSolve::memory::DEVICE));
      mem_.deviceSynchronize();
    }
    RESOLVE_RANGE_POP(__FUNCTION__);
    return error_sum;
  }

  int LinSolverDirectRocSolverRf::setSolveMode(int mode)
  {
    solve_mode_ = mode;
    return 0;
  }

  int LinSolverDirectRocSolverRf::getSolveMode()
  {
    return solve_mode_;
  }

  void LinSolverDirectRocSolverRf::addFactors(matrix::Sparse* L, matrix::Sparse* U)
  {
    // L and U need to be in CSC format
    index_type n = L->getNumRows();
    index_type* Lp = L->getColData(ReSolve::memory::HOST); 
    index_type* Li = L->getRowData(ReSolve::memory::HOST); 
    index_type* Up = U->getColData(ReSolve::memory::HOST); 
    index_type* Ui = U->getRowData(ReSolve::memory::HOST); 
    if (M_ != nullptr) {
      delete M_;
    }

    index_type nnzM = ( L->getNnz() + U->getNnz() - n );
    M_ = new matrix::Csr(n, n, nnzM);
    M_->allocateMatrixData(ReSolve::memory::DEVICE);
    M_->allocateMatrixData(ReSolve::memory::HOST);
    index_type* mia = M_->getRowData(ReSolve::memory::HOST);
    index_type* mja = M_->getColData(ReSolve::memory::HOST);
    index_type row;
    for(index_type i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(index_type j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        // BUT dont count diagonal twice, important
        if(row != i) {
          mia[row + 1]++;
        }
      }
      // then each column of U
      for(index_type j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mia[row + 1]++;
      }
    }
    // then organize mia_;
    mia[0] = 0;
    for(index_type i = 1; i < n + 1; i++) {
      mia[i] += mia[i - 1];
    }

    std::vector<int> Mshifts(static_cast<size_t>(n), 0);
    for(index_type i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        if(row != i) {
          // place (row, i) where it belongs!
          mja[mia[row] + Mshifts[static_cast<size_t>(row)]] = i;
          Mshifts[static_cast<size_t>(row)]++;
        }
      }
      // each column of U next
      for(index_type j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mja[mia[row] + Mshifts[static_cast<size_t>(row)]] = i;
        Mshifts[static_cast<size_t>(row)]++;
      }
    }
  } // LinSolverDirectRocSolverRf::addFactors
} // namespace resolve
