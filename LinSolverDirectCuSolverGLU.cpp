#include "LinSolverDirectCuSolverGLU.hpp"
#include <vector>

namespace ReSolve
{
  LinSolverDirectCuSolverGLU::LinSolverDirectCuSolverGLU(LinAlgWorkspace* workspace)
  {
    this->workspace_ = workspace;
  }

  LinSolverDirectCuSolverGLU::~LinSolverDirectCuSolverGLU()
  {
    cudaFree(glu_buffer_);
    cusparseDestroyMatDescr(descr_M_);
    cusparseDestroyMatDescr(descr_A_);
    cusolverSpDestroyGluInfo(info_M_);
    delete M_;
  }

  void LinSolverDirectCuSolverGLU::setup(Matrix* A, Matrix* L, Matrix* U, Int* P, Int* Q)
  {
    LinAlgWorkspaceCUDA* workspaceCUDA = (LinAlgWorkspaceCUDA*) workspace_;
    //get the handle
    handle_cusolversp_ = workspaceCUDA->getCusolverSpHandle();

    A_ = A;
    Int n = A_->getNumRows();
    Int nnz = A_->getNnzExpanded();
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
                                           A_->getCsrRowPointers("cpu"), //kRowPtr_,
                                           A_->getCsrColIndices("cpu"), //jCol_, 
                                           P, /* base-0 */
                                           Q,   /* base-0 */
                                           M_->getNnz(),           /* nnzM */
                                           descr_M_, 
                                           M_->getCsrRowPointers("cpu"), 
                                           M_->getCsrColIndices("cpu"), 
                                           info_M_);
    
    //NOW the buffer 
    size_t buffer_size;
    cusolverSpDgluBufferSize(handle_cusolversp_, info_M_, &buffer_size);

    cudaMalloc((void**)&glu_buffer_, buffer_size);

    cusolverSpDgluAnalysis(handle_cusolversp_, info_M_, glu_buffer_);

    // reset and refactor so factors are ON THE GPU

    cusolverSpDgluReset(handle_cusolversp_, 
                        n,
                        /* A is original matrix */
                        nnz, 
                        descr_A_, 
                        A_->getCsrValues("cuda"),  //da_, 
                        A_->getCsrRowPointers("cuda"), //kRowPtr_,
                        A_->getCsrColIndices("cuda"), //jCol_, 
                        info_M_);

    cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
  }

  void LinSolverDirectCuSolverGLU::addFactors(Matrix* L, Matrix* U)
  {
    Int n = L->getNumRows();
    Int* Lp = L->getCscColPointers("cpu"); 
    Int* Li = L->getCscRowIndices("cpu"); 
    Int* Up = U->getCscColPointers("cpu"); 
    Int* Ui = U->getCscRowIndices("cpu"); 

    Int nnzM = ( L->getNnz() + U->getNnz() - n );
    M_ = new Matrix(n, n, nnzM);
    M_->allocateCsr("cpu");

    Int* mia = M_->getCsrRowPointers("cpu");
    Int* mja = M_->getCsrColIndices("cpu");

    Int row;
    for(Int i = 0; i < n; ++i) {
      // go through EACH COLUMN OF L first
      for(Int j = Lp[i]; j < Lp[i + 1]; ++j) {
        row = Li[j];
        // BUT dont count diagonal twice, important
        if(row != i) {
          mia[row + 1]++;
        }
      }
      // then each column of U
      for(Int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mia[row + 1]++;
      }
    }
    // then organize mia_;
    mia[0] = 0;
    for(Int i = 1; i < n + 1; i++) {
      mia[i] += mia[i - 1];
    }

    std::vector<int> Mshifts(n, 0);
    for(Int i = 0; i < n; ++i) {
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
      for(Int j = Up[i]; j < Up[i + 1]; ++j) {
        row = Ui[j];
        mja[mia[row] + Mshifts[row]] = i;
        Mshifts[row]++;
      }
    }
    //Mshifts.~vector(); 
  }

  int LinSolverDirectCuSolverGLU::refactorize()
  {
    status_cusolver_ =  cusolverSpDgluReset(handle_cusolversp_, 
                                            A_->getNumRows(),
                                            /* A is original matrix */
                                            A_->getNnzExpanded(),
                                            descr_A_,
                                            A_->getCsrValues("cuda"),  //da_, 
                                            A_->getCsrRowPointers("cuda"), //kRowPtr_,
                                            A_->getCsrColIndices("cuda"), //jCol_, 
                                            info_M_);
    status_cusolver_ =  cusolverSpDgluFactor(handle_cusolversp_, info_M_, glu_buffer_);
    return status_cusolver_;
  }

  int LinSolverDirectCuSolverGLU::solve(Vector* rhs, Vector* x)
  {

    status_cusolver_ =  cusolverSpDgluSolve(handle_cusolversp_,
                                            A_->getNumRows(),
                                            /* A is original matrix */
                                            A_->getNnz(),
                                            descr_A_,
                                            A_->getCsrValues("cuda"),  //da_, 
                                            A_->getCsrRowPointers("cuda"), //kRowPtr_,
                                            A_->getCsrColIndices("cuda"), //jCol_, 
                                            rhs->getData("cuda"),/* right hand side */
                                            x->getData("cuda"),/* left hand side */
                                            &ite_refine_succ_,
                                            &r_nrminf_,
                                            info_M_,
                                            glu_buffer_);
    return status_cusolver_; 
  }

}
