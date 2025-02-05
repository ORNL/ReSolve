#pragma once

#include <iostream>
#include <resolve/LinSolverIterative.hpp>

void isTestPass(int error_sum, const std::string& test_name)
{
  using namespace ReSolve::colors;

  if (error_sum == 0) {
    std::cout << std::endl << test_name
              << GREEN << " PASSED" << CLEAR << std::endl << std::endl;
  } else {
    std::cout << std::endl << test_name
              << RED << " FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl << std::endl;
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
     * @brief Default constructor
     * 
     * Initializes matrix and vector handlers.
     * 
     * @param[in,out] workspace - workspace for matrix and vector handlers
     */
    TestHelper(workspace_type& workspace)
      : mh_(&workspace),
        vh_(&workspace)
    {
      if (mh_.getIsCudaEnabled() || mh_.getIsHipEnabled()) {
        memspace_ = ReSolve::memory::DEVICE;
      }
    }

    /**
     * @brief TestHelper constructor
     * 
     * @param A 
     * @param r 
     * @param x 
     * @param[in,out] workspace - workspace for matrix and vector handlers
     * 
     * @pre The linear solver has solved system A * x = r.
     * @pre A, r, and x are all in the same memory space as the workspace.
     * @pre Workspace handles are initialized
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
        res_(new ReSolve::vector::Vector(A->getNumRows())),
        x_true_(new ReSolve::vector::Vector(A->getNumRows()))
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

    void setSystem(ReSolve::matrix::Sparse* A,
                   ReSolve::vector::Vector* r,
                   ReSolve::vector::Vector* x)
    {
      assert((res_ == nullptr) && (x_true_ == nullptr));
      A_ = A;
      r_ = r;
      x_ = x;
      res_ = new ReSolve::vector::Vector(A->getNumRows());
      x_true_ = new ReSolve::vector::Vector(A->getNumRows());
      setSolutionVector();
      computeNorms();
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

    void setTestName(const std::string& name)
    {
      test_name_ += name;
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
      std::cout << "\t Residual norm           ||b-A*x||               : " << getNormResidual()       << "\n";
      if (memspace_ == ReSolve::memory::DEVICE) {
        std::cout << "\t Residual norm on CPU    ||b-A*x|| (CPU)         : " << getNormResidualCpu()    << "\n";
      }
      std::cout << "\t Relative residual norm  ||b-A*x||/||b||         : " << getNormResidualScaled() << "\n";
      std::cout << "\t Error norm              ||x-x_true||            : " << getNormDiff()           << "\n";
      std::cout << "\t Relative error norm     ||x-x_true||/||x_true|| : " << getNormDiffScaled()     << "\n";
      std::cout << "\t Exact solution residual ||b-A*x_true||          : " << getNormResidualTrue()   << "\n";
    }

    void printIrSummary(ReSolve::LinSolverIterative* ls)
    {
      using namespace ReSolve;

      real_type tol = ls->getTol();
      index_type maxit = ls->getMaxit();

      std::cout << std::setprecision(16) << std::scientific;
      std::cout << "\t IR initial residual norm ||b-A*x||              : " << ls->getInitResidualNorm() << "\n";
      std::cout << "\t IR final residual norm   ||b-A*x||              : " << ls->getFinalResidualNorm() << "\n";
      std::cout << "\t IR iterations                                   : " << ls->getNumIter() << "\n";
      std::cout << "\t IR tolerance                                    : " << std::setprecision(2) << tol << "\n";
      std::cout << "\t IR max iterations                               : " << maxit << "\n";
    }

    void printIterativeSolverSummary(ReSolve::LinSolverIterative* ls)
    {
      std::cout << std::setprecision(16) << std::scientific;
      std::cout << "\t IR initial residual norm          ||b-A*x||       : " << ls->getInitResidualNorm() << "\n";
      std::cout << "\t IR initial relative residual norm ||b-A*x||/||b|| : " << ls->getInitResidualNorm()/norm_rhs_ << "\n";
      std::cout << "\t IR final residual norm            ||b-A*x||       : " << ls->getFinalResidualNorm() << "\n";
      std::cout << "\t IR final relative residual norm   ||b-A*x||/||b|| : " << ls->getFinalResidualNorm()/norm_rhs_ << "\n";
      std::cout << "\t IR iterations                                     : " << ls->getNumIter() << "\n";
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

    int checkNormOfScaledResiduals(ReSolve::real_type nsr_system)
    {
      using namespace ReSolve;
      int error_sum = 0;
      
      // Compute residual norm to get updated vector res_
      res_->copyDataFrom(r_, memspace_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

      // Compute norm of scaled residuals
      real_type inf_norm_A = 0.0;
      mh_.matrixInfNorm(A_, &inf_norm_A, memspace_); 
      real_type inf_norm_x = vh_.infNorm(x_, memspace_);
      real_type inf_norm_res = vh_.infNorm(res_, memspace_);
      real_type nsr_norm   = inf_norm_res / (inf_norm_A * inf_norm_x);
      real_type error      = std::abs(nsr_system - nsr_norm)/nsr_norm;

      // Test norm of scaled residuals method in SystemSolver
      if (error > 10.0*std::numeric_limits<real_type>::epsilon()) 
      {
        std::cout << "Norm of scaled residuals computation failed:\n";
        std::cout << std::scientific << std::setprecision(16)
                  << "\tMatrix inf  norm                 : " << inf_norm_A << "\n"
                  << "\tResidual inf norm                : " << inf_norm_res << "\n"  
                  << "\tSolution inf norm                : " << inf_norm_x << "\n"  
                  << "\tNorm of scaled residuals         : " << nsr_norm   << "\n"
                  << "\tNorm of scaled residuals (system): " << nsr_system << "\n\n";
      }
      return error_sum;
    }

    int checkRelativeResidualNorm(ReSolve::real_type rrn_system)
    {
      using namespace ReSolve;
      int error_sum = 0;
      
      // Compute residual norm
      res_->copyDataFrom(r_, memspace_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

      real_type error = std::abs(norm_rhs_ * rrn_system - norm_res_)/norm_res_;
      if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
        std::cout << "Relative residual norm computation failed:\n";
        std::cout << std::scientific << std::setprecision(16)
                  << "\tTest value            : " << norm_res_/norm_rhs_ << "\n"
                  << "\tSystemSolver computed : " << rrn_system        << "\n\n";
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

      // Compute rhs and residual norms
      res_->copyDataFrom(r_, memspace_, memspace_);
      norm_rhs_ = norm2(*r_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

      // Compute residual norm w.r.t. true solution
      res_->copyDataFrom(r_, memspace_, memspace_);
      norm_res_true_ = computeResidualNorm(*A_, *x_true_, *res_, memspace_);

      // Compute residual norm on CPU
      if (memspace_ == ReSolve::memory::DEVICE) {
        A_->syncData(ReSolve::memory::HOST);
        r_->syncData(ReSolve::memory::HOST);
        x_->syncData(ReSolve::memory::HOST);
        res_->copyDataFrom(r_, memspace_, ReSolve::memory::HOST);
        norm_res_cpu_ = computeResidualNorm(*A_, *x_, *res_, ReSolve::memory::HOST);
      }

      // Compute vector difference norm
      res_->copyDataFrom(x_, memspace_, memspace_);
      norm_diff_ = computeDiffNorm(*x_true_, *res_, memspace_);
    }

    void setSolutionVector()
    {
      x_true_->allocate(memspace_);
      x_true_->setToConst(static_cast<ReSolve::real_type>(1.0), memspace_);
      x_true_->setDataUpdated(memspace_);
      x_true_->syncData(ReSolve::memory::HOST);
      norm_true_ = norm2(*x_true_, memspace_);
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

    std::string test_name_{"Test "};

    ReSolve::MatrixHandler mh_;
    ReSolve::VectorHandler vh_;

    ReSolve::vector::Vector* res_{nullptr};
    ReSolve::vector::Vector* x_true_{nullptr};

    ReSolve::real_type norm_rhs_{0.0};
    ReSolve::real_type norm_res_{0.0};
    ReSolve::real_type norm_res_cpu_{0.0};
    ReSolve::real_type norm_res_true_{0.0};
    ReSolve::real_type norm_true_{0.0};
    ReSolve::real_type norm_diff_{0.0};

    bool solution_set_{false};

    ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
};
