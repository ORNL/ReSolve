#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
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
    mem_.deleteOnDevice(d_P_);
    mem_.deleteOnDevice(d_Q_);
    mem_.deleteOnDevice(d_T_);
  }

  int LinSolverDirectCuSolverRf::setup(matrix::Sparse* A, matrix::Sparse* L, matrix::Sparse* U, index_type* P, index_type* Q)
  {
    //remember - P and Q are generally CPU variables
    int error_sum = 0;
    this->A_ = (matrix::Csr*) A;
    index_type n = A_->getNumRows();
    mem_.allocateArrayOnDevice(&d_P_, n); 
    mem_.allocateArrayOnDevice(&d_Q_, n);
    mem_.allocateArrayOnDevice(&d_T_, n);

    mem_.copyArrayHostToDevice(d_P_, P, n);
    mem_.copyArrayHostToDevice(d_Q_, Q, n);


    status_cusolverrf_ = cusolverRfSetResetValuesFastMode(handle_cusolverrf_, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    error_sum += status_cusolverrf_;
    status_cusolverrf_ = cusolverRfSetupDevice(n, 
                                               A_->getNnzExpanded(),
                                               A_->getRowData(memory::DEVICE), //dia_,
                                               A_->getColData(memory::DEVICE), //dja_,
                                               A_->getValues( memory::DEVICE), //da_,
                                               L->getNnz(),
                                               L->getRowData(memory::DEVICE),
                                               L->getColData(memory::DEVICE),
                                               L->getValues( memory::DEVICE),
                                               U->getNnz(),
                                               U->getRowData(memory::DEVICE),
                                               U->getColData(memory::DEVICE),
                                               U->getValues( memory::DEVICE),
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    mem_.deviceSynchronize();
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
                                               A_->getRowData(memory::DEVICE), //dia_,
                                               A_->getColData(memory::DEVICE), //dja_,
                                               A_->getValues( memory::DEVICE), //da_,
                                               d_P_,
                                               d_Q_,
                                               handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    mem_.deviceSynchronize();
    status_cusolverrf_ =  cusolverRfRefactor(handle_cusolverrf_);
    error_sum += status_cusolverrf_;

    return error_sum; 
  }

  // solution is returned in RHS
  int LinSolverDirectCuSolverRf::solve(vector_type* rhs)
  {
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          rhs->getData(memory::DEVICE),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::solve(vector_type* rhs, vector_type* x)
  {
    x->update(rhs->getData(memory::DEVICE), memory::DEVICE, memory::DEVICE);
    x->setDataUpdated(memory::DEVICE);
    status_cusolverrf_ =  cusolverRfSolve(handle_cusolverrf_,
                                          d_P_,
                                          d_Q_,
                                          1,
                                          d_T_,
                                          A_->getNumRows(),
                                          x->getData(memory::DEVICE),
                                          A_->getNumRows());
    return status_cusolverrf_;
  }

  int LinSolverDirectCuSolverRf::setNumericalProperties(double nzero, double nboost)
  {
    status_cusolverrf_ = cusolverRfSetNumericProperties(handle_cusolverrf_, nzero, nboost);
      return status_cusolverrf_;
  }
}// namespace resolve
