#include "LinSolverDirectCuSolverRf.hpp"

namespace ReSolve 
{
  resolveLinSolverDirectCuSolverRf::resolveLinSolverDirectCuSolverRf()
  {
    cusolverRfCreate(&handle_cusolverrf_);
  }

  resolveLinSolverDirectCuSolverRf::~resolveLinSolverDirectCuSolverRf()
  {
    cusolverRfDestroy(handle_cusolverrf_);
    cudaFree(d_P_);
    cudaFree(d_Q_);
    cudaFree(d_T_);
  }

  void resolveLinSolverDirectCuSolverRf::setup(resolveMatrix* A, resolveMatrix* L, resolveMatrix* U, resolveInt* P, resolveInt* Q)
  {
    //remember - P and Q are generally CPU variables

    resolveInt n = A_->getNumRows();

    cudaMalloc(&d_P_, n * sizeof(resolveInt)); 
    cudaMalloc(&d_Q_, n * sizeof(resolveInt));
    cudaMalloc(&d_T_, n * sizeof(resolveReal));

    cudaMemcpy(d_P_, P, n  * sizeof(resolveInt), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q_, Q, n  * sizeof(resolveInt), cudaMemcpyHostToDevice);


    status_cusolverrf_ = cusolverRfSetResetValuesFastMode(handle_cusolverrf_, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);

    cusolverRfSetupDevice(n, 
                          A_->getNnzExpanded(),
                          A_->getCsrRowPointers("cuda"), //dia_,
                          A_->getCsrColIndices("cuda"), //dja_,
                          A_->getCsrValues("cuda"),  //da_,
                          L->getNnz(),
                          L->getCsrRowPointers("cuda"),
                          L->getCsrColIndices("cuda"),
                          L->getCsrValues("cuda"),
                          U->getNnz(),
                          U->getCsrRowPointers("cuda"),
                          U->getCsrColIndices("cuda"),
                          U->getCsrValues("cuda"),
                          d_P_,
                          d_Q_,
                          handle_cusolverrf_);

    cudaDeviceSynchronize();
    cusolverRfAnalyze(handle_cusolverrf_);

    this->A_ = A;
    //default

    const cusolverRfFactorization_t fact_alg =
      CUSOLVERRF_FACTORIZATION_ALG0;  // 0 - default, 1 or 2
    const cusolverRfTriangularSolve_t solve_alg =
      CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;  //  1- default, 2 or 3 // 1 causes error
    this->setAlgorithms(fact_alg, solve_alg);
  }

  void resolveLinSolverDirectCuSolverRf::setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg)
  {
    cusolverRfSetAlgs(handle_cusolverrf_, fact_alg, solve_alg);
  }

  int resolveLinSolverDirectCuSolverRf::refactorize()
  {
    status_cusolverrf_ = cusolverRfResetValues(A_->getNumRows(), 
                                               A_->getNnzExpanded(), 
                                               A_->getCsrRowPointers("cuda"), //dia_,
                                               A_->getCsrColIndices("cuda"), //dja_,
                                               A_->getCsrValues("cuda"),  //da_,
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    cudaDeviceSynchronize();
    status_cusolverrf_ =  cusolverRfRefactor(handle_cusolverrf_);
    return status_cusolverrf_; 
  }

  int resolveLinSolverDirectCuSolverRf::solve(resolveVector* rhs, resolveVector* x)
  {
    x->update(rhs->getData("cpu"), "cpu", "cuda");
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
}
