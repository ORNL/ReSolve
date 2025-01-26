#pragma once

#include <iostream>

void isTestPass(int error_sum)
{
  using namespace ReSolve::colors;

  if (error_sum == 0) {
    std::cout << "Test KLU with rocsolverRf refactorization "
              << GREEN << "PASSED" << CLEAR << std::endl;
  } else {
    std::cout << "Test KLU with rocsolverRf refactorization "
              << RED << "FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl;
  }
}

/**
 * @brief Test helper class template
 * 
 * @tparam workspace_type 
 */
template <class workspace_type>
class TestHelper
{    
  public:
    /**
     * @brief Test Helper constructor
     * 
     * @param A 
     * @param r 
     * @param x 
     * @param workspace
     * 
     * @pre The linear solver has solved system A * x = r.
     * @pre A, r, and x are all in the same memory space as the workspace.
     */
    TestHelper(ReSolve::matrix::Sparse* A,
               ReSolve::vector::Vector* r,
               ReSolve::vector::Vector* x,
               workspace_type& workspace)
      : A_(A),
        r_(r),
        x_(x),
        mh_(&workspace),
        vh_(&workspace),
        res_(A->getNumRows()),
        x_true_(A->getNumRows())
    {
      if (mh_.getIsCudaEnabled() || mh_.getIsHipEnabled()) {
        memspace_ = ReSolve::memory::DEVICE;
      }

      setSolutionVector();
      computeNorms();
    }

    ~TestHelper()
    {
      // empty
    }

    void resetSystem(ReSolve::matrix::Sparse* A,
                     ReSolve::vector::Vector* r,
                     ReSolve::vector::Vector* x)
    {
      assert(A_->getNumRows() == A->getNumRows());
      A_ = A;
      r_ = r;
      x_ = x;
      computeNorms();
    }

    ReSolve::real_type getNormResidual()
    {
      return norm_res_;
    }

    ReSolve::real_type getNormResidualScaled()
    {
      return norm_res_/norm_rhs_;
    }

    ReSolve::real_type getNormResidualCpu()
    {
      return norm_res_cpu_;
    }

    ReSolve::real_type getNormResidualTrue()
    {
      return norm_res_true_;
    }

    ReSolve::real_type getNormDiff()
    {
      return norm_diff_;
    }

    ReSolve::real_type getNormDiffScaled()
    {
      return norm_diff_/norm_true_;
    }

    void printSummary()
    {
      std::cout << std::setprecision(16) << std::scientific;
      std::cout << "\t ||b-A*x||               : " << getNormResidual()       << " (residual norm)\n";
      if (memspace_ == ReSolve::memory::DEVICE) {
        std::cout << "\t ||b-A*x|| (CPU)         : " << getNormResidualCpu()    << " (residual norm on CPU)\n";
      }
      std::cout << "\t ||b-A*x||/||b||         : " << getNormResidualScaled() << " (scaled residual norm)\n";
      std::cout << "\t ||x-x_true||            : " << getNormDiff()           << " (solution error)\n";
      std::cout << "\t ||x-x_true||/||x_true|| : " << getNormDiffScaled()     << " (scaled solution error)\n";
      std::cout << "\t ||b-A*x_true||          : " << getNormResidualTrue()   << " (residual norm with exact solution)\n\n";
    }

    int checkResult(ReSolve::real_type tolerance)
    {
      int error_sum = 0;
      ReSolve::real_type norm = norm_res_/norm_rhs_;

      if (!std::isfinite(norm)) {
        std::cout << "Result is not a finite number!\n";
        error_sum++;
      }
      if (norm > tolerance) {
        std::cout << "Result inaccurate!\n";
        error_sum++;
      }

      return error_sum;
    }

  private:
    void computeNorms()
    {
      if (!solution_set_) {
        setSolutionVector();
      }

      // Compute rs and residual norms
      res_.copyDataFrom(r_, memspace_, memspace_);
      norm_rhs_ = norm2(*r_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, res_, memspace_);

      // Compute residual norm w.r.t. true solution
      res_.copyDataFrom(r_, memspace_, memspace_);
      norm_res_true_ = computeResidualNorm(*A_, x_true_, res_, memspace_);

      // Compute residual norm on CPU
      if (memspace_ == ReSolve::memory::DEVICE) {
        A_->syncData(ReSolve::memory::HOST);
        r_->syncData(ReSolve::memory::HOST);
        x_->syncData(ReSolve::memory::HOST);
        res_.copyDataFrom(r_, memspace_, ReSolve::memory::HOST);
        norm_res_cpu_ = computeResidualNorm(*A_, *x_, res_, ReSolve::memory::HOST);
      }

      // Compute vector difference norm
      res_.copyDataFrom(x_, memspace_, memspace_);
      norm_diff_ = computeDiffNorm(x_true_, res_, memspace_);
    }

    void setSolutionVector()
    {
      x_true_.allocate(memspace_);
      x_true_.setToConst(static_cast<ReSolve::real_type>(1.0), memspace_);
      x_true_.setDataUpdated(memspace_);
      x_true_.syncData(ReSolve::memory::HOST);
      norm_true_ = norm2(x_true_, memspace_);
      solution_set_ = true;
    }

    /**
     * @brief Computes residual norm = || A * x - r ||_2
     * 
     * @param[in]     A 
     * @param[in]     x 
     * @param[in,out] r 
     * @return ReSolve::real_type 
     * 
     * @post r is overwritten with residual values
     */
    ReSolve::real_type computeResidualNorm(ReSolve::matrix::Sparse& A,
                                           ReSolve::vector::Vector& x,
                                           ReSolve::vector::Vector& r,
                                           ReSolve::memory::MemorySpace memspace)
    {
      using namespace ReSolve::constants;
      mh_.matvec(&A, &x, &r, &ONE, &MINUSONE, memspace); // r := A * x - r
      return norm2(r, memspace);
    }

    /**
     * @brief Compute vector difference norm = || x - x_true ||_2
     * 
     * @param[in]     x_true 
     * @param[in,out] x 
     * @param[in]     memspace 
     * @return ReSolve::real_type
     * 
     * @post x is overwritten with difference value
     */
    ReSolve::real_type computeDiffNorm(ReSolve::vector::Vector& x_true,
                                       ReSolve::vector::Vector& x,
                                       ReSolve::memory::MemorySpace memspace)
    {
      using namespace ReSolve::constants;
      vh_.axpy(&MINUSONE, &x_true, &x, memspace); // x := -x_true + x
      return norm2(x, memspace);
    }

    ReSolve::real_type norm2(ReSolve::vector::Vector& r,
                             ReSolve::memory::MemorySpace memspace)
    {
      return std::sqrt(vh_.dot(&r, &r, memspace));
    }

  private:
    ReSolve::matrix::Sparse* A_;
    ReSolve::vector::Vector* r_;
    ReSolve::vector::Vector* x_;

    ReSolve::MatrixHandler mh_;
    ReSolve::VectorHandler vh_;

    ReSolve::vector::Vector res_;
    ReSolve::vector::Vector x_true_;

    ReSolve::real_type norm_rhs_{0.0};
    ReSolve::real_type norm_res_{0.0};
    ReSolve::real_type norm_res_cpu_{0.0};
    ReSolve::real_type norm_res_true_{0.0};
    ReSolve::real_type norm_true_{0.0};
    ReSolve::real_type norm_diff_{0.0};

    bool solution_set_{false};

    ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
};
