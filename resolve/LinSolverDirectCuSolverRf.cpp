#include "LinSolverDirectCuSolverRf.hpp"

namespace ReSolve 
{
  LinSolverDirectCuSolverRf::LinSolverDirectCuSolverRf()
  {
    cusolverRfCreate(&handle_cusolverrf_);
  }

  LinSolverDirectCuSolverRf::~LinSolverDirectCuSolverRf()
  {
    cusolverRfDestroy(handle_cusolverrf_);
    cudaFree(d_P_);
    cudaFree(d_Q_);
    cudaFree(d_T_);
  }

  int LinSolverDirectCuSolverRf::setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q)
  {
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    cudaMalloc(&d_P_, n * sizeof(index_type)); 
    cudaMalloc(&d_Q_, n * sizeof(index_type));
    cudaMalloc(&d_T_, n * sizeof(real_type));

    cudaMemcpy(d_P_, P, n  * sizeof(index_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q_, Q, n  * sizeof(index_type), cudaMemcpyHostToDevice);


    status_cusolverrf_ = cusolverRfSetResetValuesFastMode(handle_cusolverrf_, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    error_sum += status_cusolverrf_;
    status_cusolverrf_ = cusolverRfSetupDevice(n, 
                                               A_->getNnzExpanded(),
                                               A_->getRowData("cuda"), //dia_,
                                               A_->getColData("cuda"), //dja_,
                                               A_->getValues("cuda"),  //da_,
                                               L->getNnz(),
                                               L->getRowData("cuda"),
                                               L->getColData("cuda"),
                                               L->getValues("cuda"),
                                               U->getNnz(),
                                               U->getRowData("cuda"),
                                               U->getColData("cuda"),
                                               U->getValues("cuda"),
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    cudaDeviceSynchronize();
    status_cusolverrf_ = cusolverRfAnalyze(handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    this->A_ = A;
    //default

    const cusolverRfFactorization_t fact_alg =
      CUSOLVERRF_FACTORIZATION_ALG0;  // 0 - default, 1 or 2
    const cusolverRfTriangularSolve_t solve_alg =
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;  //  1- default, 2 or 3 // 1 causes error
    this->setAlgorithms(fact_alg, solve_alg);
    return error_sum;
  }

  void LinSolverDirectCuSolverRf::setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg)
  {
    cusolverRfSetAlgs(handle_cusolverrf_, fact_alg, solve_alg);
  }

  int LinSolverDirectCuSolverRf::refactorize()
  {
    int error_sum = 0;
    status_cusolverrf_ = cusolverRfResetValues(A_->getNumRows(), 
                                               A_->getNnzExpanded(), 
                                               A_->getRowData("cuda"), //dia_,
                                               A_->getColData("cuda"), //dja_,
                                               A_->getValues("cuda"),  //da_,
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    cudaDeviceSynchronize();
    status_cusolverrf_ =  cusolverRfRefactor(handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectCuSolverRf::solve(Vector* rhs)
  {
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          rhs->getData("cuda"),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::solve(Vector* rhs, Vector* x)
  {
    x->update(rhs->getData("cuda"), "cuda", "cuda");
    x->setDataUpdated("cuda");
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          x->getData("cuda"),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }
} // namespace ReSolve 
