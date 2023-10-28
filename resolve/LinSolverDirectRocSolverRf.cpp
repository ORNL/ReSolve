#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include "LinSolverDirectRocSolverRf.hpp"

namespace ReSolve 
{
  LinSolverDirectRocSolverRf::LinSolverDirectRocSolverRf(LinAlgWorkspaceHIP* workspace)
  {
    workspace_ = workspace;
    infoM_ = nullptr;
    solve_mode_ = 0; //solve mode - slow mode is default
  }

  LinSolverDirectRocSolverRf::~LinSolverDirectRocSolverRf()
  {
    mem_.deleteOnDevice(d_P_);
    mem_.deleteOnDevice(d_Q_);
    mem_.deleteOnDevice(d_T_);
  }

  int LinSolverDirectRocSolverRf::setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q, vector_type* rhs)
  {
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    index_type nnz = A_->getNnzExpanded();  
    //set matrix info
    rocsolver_create_rfinfo(&infoM_, workspace_->getRocblasHandle()); 
    //create combined factor
    addFactors(L,U);

    mem_.allocateArrayOnDevice(&d_P_, n); 
    mem_.allocateArrayOnDevice(&d_Q_, n);
    mem_.allocateArrayOnDevice(&d_T_, n);

    mem_.copyArrayHostToDevice(d_P_, P, n);
    mem_.copyArrayHostToDevice(d_Q_, Q, n);


    status_rocblas_ = rocsolver_dcsrrf_analysis(workspace_->getRocblasHandle(),
                                                n,
                                                1,
                                                nnz,
                                                A_->getRowData("hip"), //kRowPtr_,
                                                A_->getColData("hip"), //jCol_, 
                                                A_->getValues("hip"), //vals_, 
                                                M_->getNnz(),
                                                M_->getRowData("hip"), 
                                                M_->getColData("hip"), 
                                                M_->getValues("hip"), //vals_, 
                                                d_P_,
                                                d_Q_,
                                                rhs->getData("hip"), 
                                                n,
                                                infoM_);

    printf("ANALYSIS status is %d \n",status_rocblas_ );
    error_sum += status_rocblas_;

    mem_.deviceSynchronize();

    this->A_ = A;

    return error_sum;
  }

  int LinSolverDirectRocSolverRf::refactorize()
  {
    int error_sum = 0;
    status_rocblas_ =  rocsolver_dcsrrf_refactlu(workspace_->getRocblasHandle(),
                                                 A_->getNumRows(),
                                                 A_->getNnzExpanded(),
                                                 A_->getRowData("hip"), //kRowPtr_,
                                                 A_->getColData("hip"), //jCol_, 
                                                 A_->getValues("hip"), //vals_, 
                                                 M_->getNnz(),
                                                 M_->getRowData("hip"), 
                                                 M_->getColData("hip"), 
                                                 M_->getValues("hip"), //OUTPUT, 
                                                 d_P_,
                                                 d_Q_,
                                                 infoM_);


    error_sum += status_rocblas_;

    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectRocSolverRf::solve(vector_type* rhs)
  {
    if (solve_mode_ == 0) {
      status_rocblas_ =  rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                                A_->getNumRows(),
                                                1,
                                                M_->getNnz(),
                                                M_->getRowData("hip"), 
                                                M_->getColData("hip"), 
                                                M_->getValues("hip"), 
                                                d_P_,
                                                d_Q_,
                                                rhs->getData("hip"),
                                                A_->getNumRows(),
                                                infoM_);
    } else {
      // not implemented yet
    }
    return status_rocblas_;
  }

  int LinSolverDirectRocSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    x->update(rhs->getData("hip"), "hip", "hip");
    x->setDataUpdated("hip");

    if (solve_mode_ == 0) {
      status_rocblas_ =  rocsolver_dcsrrf_solve(workspace_->getRocblasHandle(),
                                                A_->getNumRows(),
                                                1,
                                                M_->getNnz(),
                                                M_->getRowData("hip"), 
                                                M_->getColData("hip"),  
                                                M_->getValues("hip"), 
                                                d_P_,
                                                d_Q_,
                                                x->getData("hip"),
                                                A_->getNumRows(),
                                                infoM_);
    } else {
      // not implemented yet
    }
    return status_rocblas_;
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
    index_type* Lp = L->getColData("cpu"); 
    index_type* Li = L->getRowData("cpu"); 
    index_type* Up = U->getColData("cpu"); 
    index_type* Ui = U->getRowData("cpu"); 

    index_type nnzM = ( L->getNnz() + U->getNnz() - n );
    M_ = new matrix::Csr(n, n, nnzM);
    M_->allocateMatrixData("cpu");
    M_->allocateMatrixData("hip");
    index_type* mia = M_->getRowData("cpu");
    index_type* mja = M_->getColData("cpu");
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

    std::vector<int> Mshifts(n, 0);
    for(index_type i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        if(row != i) {
          // place (row, i) where it belongs!
          mja[mia[row] + Mshifts[row]] = i;
          Mshifts[row]++;
        }
      }
      // each column of U next
      for(index_type j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mja[mia[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    //Mshifts.~vector(); 
  }
}// namespace resolve
