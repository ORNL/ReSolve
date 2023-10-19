#include <cstring> // includes memcpy
#include <vector>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include "LinSolverDirectCuSolverGLU.hpp"

namespace ReSolve
{
  LinSolverDirectCuSolverGLU::LinSolverDirectCuSolverGLU(LinAlgWorkspaceCUDA* workspace)
  {
    this->workspace_ = workspace;
  }

  LinSolverDirectCuSolverGLU::~LinSolverDirectCuSolverGLU()
  {
    mem_.deleteOnDevice(glu_buffer_);
    cusparseDestroyMatDescr(descr_M_);
    cusparseDestroyMatDescr(descr_A_);
    cusolverSpDestroyGluInfo(info_M_);
    delete M_;
  }

  int LinSolverDirectCuSolverGLU::setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q)
  {
    int error_sum = 0;

    LinAlgWorkspaceCUDA* workspaceCUDA = workspace_;
    //get the handle
    handle_cusolversp_ = workspaceCUDA->getCusolverSpHandle();
    A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    index_type nnz = A_->getNnzExpanded();
    //create combined factor
    addFactors(L,U);

    //set up descriptors
    cusparseCreateMatDescr(&descr_M_);
    cusparseSetMatType(descr_M_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_M_, CUSPARSE_INDEX_BASE_ZERO);
    cusolverSpCreateGluInfo(&info_M_);

    cusparseCreateMatDescr(&descr_A_);
    cusparseSetMatType(descr_A_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_A_, CUSPARSE_INDEX_BASE_ZERO);

    //set up the GLU 
    status_cusolver_ = cusolverSpDgluSetup(handle_cusolversp_, 
                                           n,
                                           nnz, 
                                           descr_A_, 
                                           A_->getRowData("cpu"), //kRowPtr_,
                                           A_->getColData("cpu"), //jCol_, 
                                           P, /* base-0 */
                                           Q,   /* base-0 */
                                           M_->getNnz(),           /* nnzM */
                                           descr_M_, 
                                           M_->getRowData("cpu"), 
                                           M_->getColData("cpu"), 
                                           info_M_);
    error_sum += status_cusolver_; 
    //NOW the buffer 
    size_t buffer_size;
    status_cusolver_ = cusolverSpDgluBufferSize(handle_cusolversp_, info_M_, &buffer_size);
    error_sum += status_cusolver_; 

    mem_.allocateBufferOnDevice(&glu_buffer_, buffer_size);

    status_cusolver_ = cusolverSpDgluAnalysis(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_; 

    // reset and refactor so factors are ON THE GPU

    status_cusolver_ = cusolverSpDgluReset(handle_cusolversp_, 
                                           n,
                                           /* A is original matrix */
                                           nnz, 
                                           descr_A_, 
                                           A_->getValues("cuda"),  //da_, 
                                           A_->getRowData("cuda"), //kRowPtr_,
                                           A_->getColData("cuda"), //jCol_, 
                                           info_M_);
    error_sum += status_cusolver_; 

    status_cusolver_ = cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_; 

    return error_sum;
  }

  void LinSolverDirectCuSolverGLU::addFactors(matrix::Sparse* L, matrix::Sparse* U)
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

  int LinSolverDirectCuSolverGLU::refactorize()
  {
    int error_sum = 0;
    status_cusolver_ =  cusolverSpDgluReset(handle_cusolversp_, 
                                            A_->getNumRows(),
                                            /* A is original matrix */
                                            A_->getNnzExpanded(),
                                            descr_A_,
                                            A_->getValues("cuda"),  //da_, 
                                            A_->getRowData("cuda"), //kRowPtr_,
                                            A_->getColData("cuda"), //jCol_, 
                                            info_M_);
    error_sum += status_cusolver_;

    status_cusolver_ =  cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
    error_sum += status_cusolver_;

    return error_sum;
  }

  int LinSolverDirectCuSolverGLU::solve(vector_type* rhs, vector_type* x)
  {

    status_cusolver_ =  cusolverSpDgluSolve(handle_cusolversp_,
                                            A_->getNumRows(),
                                            /* A is original matrix */
                                            A_->getNnz(),
                                            descr_A_,
                                            A_->getValues("cuda"),  //da_, 
                                            A_->getRowData("cuda"), //kRowPtr_,
                                            A_->getColData("cuda"), //jCol_, 
                                            rhs->getData("cuda"),/* right hand side */
                                            x->getData("cuda"),/* left hand side */
                                            &ite_refine_succ_,
                                            &r_nrminf_,
                                            info_M_,
                                            glu_buffer_);
    return status_cusolver_; 
  }

}
