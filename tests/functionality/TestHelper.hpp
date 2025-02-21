#pragma once

#include <iostream>
#include <resolve/LinSolverIterative.hpp>

/**
 * @brief Checks the error code and prints pass/fail message.
 * 
 * @param error_sum - error code: 0 = pass, otherwise fail
 * @param test_name - test name to be displayed with pass/fail message
 */
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
 * This is header-only implementation of several utility functions used by
 * multiple functionality tests, such as error norm calculations. To use,
 * simply include this header in the test.
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
     * 
     * @pre Workspace handles are initialized
     * 
     * @post Handlers are instantiated.
     * allocated
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
     * @param A[in] - Linear system matrix
     * @param r[in] - Linear system right-hand side
     * @param x[in] - Computed solution of the linear system
     * @param[in,out] workspace - workspace for matrix and vector handlers
     * 
     * @pre The linear solver has solved system A * x = r.
     * @pre A, r, and x are all in the same memory space as the workspace.
     * @pre Workspace handles are initialized
     * 
     * @post Handlers are instantiated and vectors res_ and x_true_ are
     * allocated
     * @post Solution vector x_true_ elements are all set to 1.
     * @post Solution error with respect to x_true_ and residual norms
     * are computed.
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

    /**
     * @brief Destroy the TestHelper object
     * 
     * @post Vectors res_ and x_true_ are deleted.
     * 
     */
    ~TestHelper()
    {
      if (res_) {
        delete res_;
        res_ = nullptr;
      }
      if (x_true_) {
        delete x_true_;
        x_true_ = nullptr;
      }
    }

    /**
     * @brief Set the new linear system together with its computed solution
     * and compute solution error and residual norms.
     * 
     * This will set the new system A*x = r and compute related error norms.
     * 
     * @param A[in] - Linear system matrix
     * @param r[in] - Linear system right-hand side
     * @param x[in] - Computed solution of the linear system
     */
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

    /**
     * @brief Set the new linear system together with its computed solution
     * and compute solution error and residual norms.
     * 
     * This is to be used after values in A and r are updated.
     * 
     * @todo This method probably does not need any input parameters.
     * 
     * @param A[in] - Linear system matrix
     * @param r[in] - Linear system right-hand side
     * @param x[in] - Computed solution of the linear system
     */
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

    /// Set the name of the test to `name`.
    void setTestName(const std::string& name)
    {
      test_name_ += name;
    }

    /// Return L2 norm of the linear system residual.
    ReSolve::real_type getNormResidual()
    {
      return norm_res_;
    }

    /// Return relative residual norm.
    ReSolve::real_type getNormResidualScaled()
    {
      return norm_res_/norm_rhs_;
    }

    /// Return L2 residual norm computed on the host.
    ReSolve::real_type getNormResidualCpu()
    {
      return norm_res_cpu_;
    }

    /// Return L2 norm of the residual computed with the "exact" solution.
    ReSolve::real_type getNormResidualTrue()
    {
      return norm_res_true_;
    }

    /// Return L2 norm of difference between computed and the "exact" solution.
    ReSolve::real_type getNormDiff()
    {
      return norm_diff_;
    }

    /// Return L2 norm of relative difference between computed and the "exact"
    /// solution.
    ReSolve::real_type getNormDiffScaled()
    {
      return norm_diff_/norm_true_;
    }

    /// Summary of error norms for a direct solver test.
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

    /// Summary of error norms for an iterative refinement test.
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

    /// Summary of error norms for an iterative solver test.
    void printIterativeSolverSummary(ReSolve::LinSolverIterative* ls)
    {
      std::cout << std::setprecision(16) << std::scientific;
      std::cout << "\t Initial residual norm          ||b-A*x||       : " << ls->getInitResidualNorm() << "\n";
      std::cout << "\t Initial relative residual norm ||b-A*x||/||b|| : " << ls->getInitResidualNorm()/norm_rhs_ << "\n";
      std::cout << "\t Final residual norm            ||b-A*x||       : " << ls->getFinalResidualNorm() << "\n";
      std::cout << "\t Final relative residual norm   ||b-A*x||/||b|| : " << ls->getFinalResidualNorm()/norm_rhs_ << "\n";
      std::cout << "\t Number of iterations                           : " << ls->getNumIter() << "\n";
    }

    /// Check the relative residual norm against `tolerance`.
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

    /**
     * @brief Verify the computation of the norm of scaled residuals.
     * 
     * The norm value is provided as the input. This function computes 
     * the norm of scaled residuals for the system that has been set
     * by the constructor or (re)setSystem functions.
     * 
     * @param nsr_system - norm of scaled residuals value to be verified 
     * @return int - 0 if the result is correct, error code otherwise
     */
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

    /**
     * @brief Verify the computation of the relative residual norm.
     * 
     * The norm value is provided as the input. This function computes 
     * the relative residual norm for the system that has been set
     * by the constructor or (re)setSystem functions.
     * 
     * @param rrn_system - relative residual norm value to be verified 
     * @return int - 0 if the result is correct, error code otherwise
     */
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

    /**
     * @brief Verify the computation of the residual norm.
     * 
     * The norm value is provided as the input. This function computes 
     * the residual norm for the system that has been set by the constructor
     * or (re)setSystem functions.
     * 
     * @param rrn_system - residual norm value to be verified 
     * @return int - 0 if the result is correct, error code otherwise
     */
    int checkResidualNorm(ReSolve::real_type rn_system)
    {
      using namespace ReSolve;
      int error_sum = 0;
      
      // Compute residual norm
      res_->copyDataFrom(r_, memspace_, memspace_);
      norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

      real_type error = std::abs(rn_system - norm_res_)/norm_res_;
      if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
        std::cout << "Residual norm computation failed:\n";
        std::cout << std::scientific << std::setprecision(16)
                  << "\tTest value            : " << norm_res_ << "\n"
                  << "\tSystemSolver computed : " << rn_system << "\n\n";
        error_sum++;
      }
      return error_sum;
    }

  private:
    /// Compute error norms.
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

    /// Sets all elements of the solution vector to 1. 
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
     * @param[in]     A - system matrix 
     * @param[in]     x - computed solution of the system
     * @param[in,out] r - system right-hand side, residual vector
     * @param[in]     memspace memory space where to computate the norm
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
      mh_.matvec(&A, &x, &r, &ONE, &MINUS_ONE, memspace); // r := A * x - r
      return norm2(r, memspace);
    }

    /**
     * @brief Compute vector difference norm = || x - x_true ||_2
     * 
     * @param[in]     x_true - The "exact" solution
     * @param[in,out] x      - Computed solution, difference vector
     * @param[in]     memspace memory space where to computate the norm
     * @return ReSolve::real_type
     * 
     * @post x is overwritten with difference value
     */
    ReSolve::real_type computeDiffNorm(ReSolve::vector::Vector& x_true,
                                       ReSolve::vector::Vector& x,
                                       ReSolve::memory::MemorySpace memspace)
    {
      using namespace ReSolve::constants;
      vh_.axpy(&MINUS_ONE, &x_true, &x, memspace); // x := -x_true + x
      return norm2(x, memspace);
    }

    /// Compute L2 norm of vector `r` in memory space `memspace`.
    ReSolve::real_type norm2(ReSolve::vector::Vector& r,
                             ReSolve::memory::MemorySpace memspace)
    {
      return std::sqrt(vh_.dot(&r, &r, memspace));
    }

  private:
    ReSolve::matrix::Sparse* A_; ///< pointer to system matrix
    ReSolve::vector::Vector* r_; ///< pointer to system right-hand side
    ReSolve::vector::Vector* x_; ///< pointer to the computed solution

    std::string test_name_{"Test "}; ///< test name

    ReSolve::MatrixHandler mh_; ///< matrix handler instance
    ReSolve::VectorHandler vh_; ///< vector handler instance

    ReSolve::vector::Vector* res_{nullptr};    ///< pointer to residual vector
    ReSolve::vector::Vector* x_true_{nullptr}; ///< pointer to solution error vector

    ReSolve::real_type norm_rhs_{0.0}; ///< right-hand side vector norm
    ReSolve::real_type norm_res_{0.0}; ///< residual vector norm
    ReSolve::real_type norm_res_cpu_{0.0}; ///< residual vector norm (on host)
    ReSolve::real_type norm_res_true_{0.0}; ///< residual norm for "exact" solution
    ReSolve::real_type norm_true_{0.0}; ///< norm of the "exact" solution
    ReSolve::real_type norm_diff_{0.0}; ///< norm of solution error

    bool solution_set_{false}; ///< if exact solution is set

    ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
};
