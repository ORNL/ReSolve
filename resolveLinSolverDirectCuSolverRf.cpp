#include "resolveLinSolverDirectCuSolverRf.hpp"

namespace ReSolve {
  resolveLinSolverDirectCuSolverRf::resolveLinSolverDirectCuSolverRf()
  {
    cusolverRfCreate(&handle_cusolverrf);
  }

  resolveLinSolverDirectCuSolverRf::~resolveLinSolverDirectCuSolverRf()
  {
  }

  void resolveLinSolverDirectCuSolverRf::setup(resolveMatrix* A, resolveMatrix* L, resolveMatrix* U, resolveInt* P, resolveInt* Q)
  {
    //remember - P and Q are generally CPU variables

    resolveInt n = A->getNumRows();


    cudaMalloc(&d_P, n * sizeof(resolveInt)); 
    cudaMalloc(&d_Q, n * sizeof(resolveInt));
    cudaMalloc(&d_T, n * sizeof(resolveReal));

    cudaMemcpy(d_P, P, n  * sizeof(resolveInt), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, Q, n  * sizeof(resolveInt), cudaMemcpyHostToDevice);

    cusolverRfSetupDevice(n, 
                          A->getNnzExpanded(),
                          A->getCsrRowPointers("cuda"), //dia_,
                          A->getCsrColIndices("cuda"), //dja_,
                          A->getCsrValues("cuda"),  //da_,
                          L->getNnz(),
                          L->getCsrRowPointers("cuda"),
                          L->getCsrColIndices("cuda"),
                          L->getCsrValues("cuda"),
                          U->getNnz(),
                          U->getCsrRowPointers("cuda"),
                          U->getCsrColIndices("cuda"),
                          U->getCsrValues("cuda"),
                          d_P,
                          d_Q,
                          handle_cusolverrf);
    cudaDeviceSynchronize();
    cusolverRfAnalyze(handle_cusolverrf);

    this->A = A;
//default

  const cusolverRfFactorization_t fact_alg =
    CUSOLVERRF_FACTORIZATION_ALG0;  // 0 - default, 1 or 2
  const cusolverRfTriangularSolve_t solve_alg =
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;  //  1- default, 2 or 3 // 1 causes error
  this->setAlgorithms(fact_alg, solve_alg);
  }

  void resolveLinSolverDirectCuSolverRf::setAlgorithms(cusolverRfFactorization_t fact_alg,  cusolverRfTriangularSolve_t solve_alg)
  {
    cusolverRfSetAlgs(handle_cusolverrf, fact_alg, solve_alg);
  }

  int resolveLinSolverDirectCuSolverRf::refactorize()
  {
   status_cusolverrf = cusolverRfResetValues(A->getNumRows(), 
                          A->getNnzExpanded(), 
                          A->getCsrRowPointers("cuda"), //dia_,
                          A->getCsrColIndices("cuda"), //dja_,
                          A->getCsrValues("cuda"),  //da_,
                          d_P,
                          d_Q,
                          handle_cusolverrf);
    cudaDeviceSynchronize();
   status_cusolverrf =  cusolverRfRefactor(handle_cusolverrf);
  return status_cusolverrf; 
  }
  int resolveLinSolverDirectCuSolverRf::solve(resolveVector* rhs, resolveVector* x)
  {

    x->update(rhs->getData("cpu"), "cuda", "cpu");
    x->setDataUpdated("cpu");
   status_cusolverrf =  cusolverRfSolve(handle_cusolverrf,
                    d_P,
                    d_Q,
                    1,
                    d_T,
                    A->getNumRows(),
                    x->getData("cuda"),
                    A->getNumRows());
 return status_cusolverrf;
  }
}
