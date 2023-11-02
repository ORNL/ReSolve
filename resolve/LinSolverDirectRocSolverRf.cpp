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
  }

  int LinSolverDirectRocSolverRf::setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q, vector_type* rhs)
  {
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    //set matrix info
    rocsolver_create_rfinfo(&infoM_, workspace_->getRocblasHandle()); 
    //create combined factor
    addFactors(L,U);
    M_->setUpdated(ReSolve::memory::HOST);
    M_->copyData(ReSolve::memory::DEVICE);
    mem_.allocateArrayOnDevice(&d_P_, n); 
    mem_.allocateArrayOnDevice(&d_Q_, n);

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


    return error_sum;
  }

  int LinSolverDirectRocSolverRf::refactorize()
  {
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

    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectRocSolverRf::solve(vector_type* rhs)
  {
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
    }
    return status_rocblas_;
  }

  int LinSolverDirectRocSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    x->update(rhs->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::DEVICE);
    x->setDataUpdated(ReSolve::memory::DEVICE);

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
      mem_.deviceSynchronize();
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
    index_type* Lp = L->getColData(ReSolve::memory::HOST); 
    index_type* Li = L->getRowData(ReSolve::memory::HOST); 
    index_type* Up = U->getColData(ReSolve::memory::HOST); 
    index_type* Ui = U->getRowData(ReSolve::memory::HOST); 

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
