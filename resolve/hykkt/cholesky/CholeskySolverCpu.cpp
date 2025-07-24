#include "CholeskySolverCpu.hpp"

namespace ReSolve {
  using real_type = ReSolve::real_type;
  using out = ReSolve::io::Logger;

  namespace hykkt {
    CholeskySolverCpu::CholeskySolverCpu() {
      Common_.nmethods = 1;
      // Use natural ordering
      Common_.method[0].ordering = CHOLMOD_NATURAL;
    }

    CholeskySolverCpu::~CholeskySolverCpu() {
      cholmod_free_sparse(&A_chol_, &Common_);
    }

    void CholeskySolverCpu::addMatrixInfo(matrix::Csr* A) {
      A_chol_ = convertToCholmod(A);
    }

    void CholeskySolverCpu::symbolicAnalysis() {
      factorization_ = cholmod_analyze(A_chol_, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky symbolic analysis failed.";
      }
    }

    void CholeskySolverCpu::numericalFactorization(real_type tol) {
      // TODO: What to do with tol for CPU?
      cholmod_factorize(A_chol_, factorization_, &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky factorization failed.";
      }
    }

    void CholeskySolverCpu::solve(vector::Vector* x, vector::Vector* b) {
      cholmod_dense* x_chol_ = cholmod_solve(CHOLMOD_A, factorization_, convertToCholmod(b), &Common_);
      if (Common_.status < 0)
      {
        out::error() << "Cholesky solve failed.";
      }
      x->copyDataFrom(static_cast<real_type*>(x_chol_->x), memory::HOST, memory::HOST);
    }

    cholmod_sparse* CholeskySolverCpu::convertToCholmod(matrix::Csr* A) {
      A_chol_ = cholmod_allocate_sparse(A->getNumRows(), A->getNumColumns(), A->getNnz(), 1, 1, 1, CHOLMOD_REAL, &Common_);
      A_chol_->p = A->getRowData(memory::HOST);
      A_chol_->i = A->getColData(memory::HOST);
      A_chol_->x = A->getValues(memory::HOST);

      return A_chol_;
    }

    cholmod_dense* CholeskySolverCpu::convertToCholmod(vector::Vector* v) {
      cholmod_dense* v_chol = cholmod_allocate_dense(v->getSize(), 1, v->getSize(), CHOLMOD_REAL, &Common_);
      v_chol->x = v->getData(memory::HOST);
      return v_chol;
    }
  }
}