#pragma once

#include <cassert>
#include <cmath>
#include <iostream>

#include <resolve/LinSolverIterative.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  namespace examples
  {
    /**
     * @brief Prints linear system info.
     *
     * @param name - pathname of the system matrix
     * @param A    - pointer to the system matrix
     */
    void printSystemInfo(const std::string& matrix_pathname, matrix::Sparse* A)
    {
      std::cout << std::endl;
      std::cout << "========================================================================================================================\n";
      std::cout << "Reading: " << matrix_pathname << std::endl;
      std::cout << "========================================================================================================================\n";
      std::cout << std::endl;

      std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x " << A->getNumColumns()
                << ", nnz: " << A->getNnz()
                << ", symmetric? " << A->symmetric()
                << ", Expanded? " << A->expanded() << std::endl;
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
    class ExampleHelper
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
      ExampleHelper(workspace_type& workspace)
        : mh_(&workspace),
          vh_(&workspace)
      {
        memspace_ = ReSolve::memory::DEVICE;
        if (mh_.getIsCudaEnabled())
        {
          hardware_backend_ = "CUDA";
        }
        else if (mh_.getIsHipEnabled())
        {
          hardware_backend_ = "HIP";
        }
        else
        {
          hardware_backend_ = "CPU";
          memspace_         = ReSolve::memory::HOST;
        }
      }

      /**
       * @brief Destroy the ExampleHelper object
       *
       * @post Vectors res_ and x_true_ are deleted.
       *
       */
      ~ExampleHelper()
      {
        if (res_)
        {
          delete res_;
          res_ = nullptr;
        }
        if (x_true_)
        {
          delete x_true_;
          x_true_ = nullptr;
        }
      }

      /// Returns the configured hardware backend
      std::string getHardwareBackend() const
      {
        return hardware_backend_;
      }

      /// Returns the configured memory space
      ReSolve::memory::MemorySpace getMemspace() const
      {
        return memspace_;
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
        A_   = A;
        r_   = r;
        x_   = x;
        res_ = new ReSolve::vector::Vector(A->getNumRows());
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
        A_ = A;
        r_ = r;
        x_ = x;
        if (res_ == nullptr)
        {
          res_ = new ReSolve::vector::Vector(A->getNumRows());
        }

        computeNorms();
      }

      /// Return L2 norm of the linear system residual.
      ReSolve::real_type getNormResidual()
      {
        return norm_res_;
      }

      /// Return relative residual norm.
      ReSolve::real_type getNormRelativeResidual()
      {
        return norm_res_ / norm_rhs_;
      }

      /// Minimalistic summary
      void printShortSummary()
      {
        std::cout << "\t2-Norm of the residual: "
                  << std::scientific << std::setprecision(16)
                  << getNormRelativeResidual() << "\n";
      }

      /// Summary of direct solve
      void printSummary()
      {
        std::cout << "\t 2-Norm of the residual (before IR): "
                  << std::scientific << std::setprecision(16)
                  << getNormRelativeResidual() << "\n";

        std::cout << std::scientific << std::setprecision(16)
                  << "\t Matrix inf  norm: " << inf_norm_A_ << "\n"
                  << "\t Residual inf norm: " << inf_norm_res_ << "\n"
                  << "\t Solution inf norm: " << inf_norm_x_ << "\n"
                  << "\t Norm of scaled residuals: " << nsr_norm_ << "\n";
      }

      /// Summary of error norms for an iterative refinement test.
      void printIrSummary(ReSolve::LinSolverIterative* ls)
      {
        std::cout << "FGMRES: init nrm: "
                  << std::scientific << std::setprecision(16)
                  << ls->getInitResidualNorm() / norm_rhs_
                  << " final nrm: "
                  << ls->getFinalResidualNorm() / norm_rhs_
                  << " iter: " << ls->getNumIter() << "\n";
      }

      /// Summary of error norms for an iterative solver test.
      void printIterativeSolverSummary(ReSolve::LinSolverIterative* ls)
      {
        std::cout << std::setprecision(16) << std::scientific;
        std::cout << "\t Initial residual norm          ||b-A*x||       : " << ls->getInitResidualNorm() << "\n";
        std::cout << "\t Initial relative residual norm ||b-A*x||/||b|| : " << ls->getInitResidualNorm() / norm_rhs_ << "\n";
        std::cout << "\t Final residual norm            ||b-A*x||       : " << ls->getFinalResidualNorm() << "\n";
        std::cout << "\t Final relative residual norm   ||b-A*x||/||b|| : " << ls->getFinalResidualNorm() / norm_rhs_ << "\n";
        std::cout << "\t Number of iterations                           : " << ls->getNumIter() << "\n";
      }

      /// Check the relative residual norm against `tolerance`.
      int checkResult(ReSolve::real_type tolerance)
      {
        int                error_sum = 0;
        ReSolve::real_type norm      = norm_res_ / norm_rhs_;

        if (!std::isfinite(norm))
        {
          std::cout << "Result is not a finite number!\n";
          error_sum++;
        }
        if (norm > tolerance)
        {
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
        real_type inf_norm_x   = vh_.infNorm(x_, memspace_);
        real_type inf_norm_res = vh_.infNorm(res_, memspace_);
        real_type nsr_norm     = inf_norm_res / (inf_norm_A * inf_norm_x);
        real_type error        = std::abs(nsr_system - nsr_norm) / nsr_norm;

        // Test norm of scaled residuals method in SystemSolver
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
          std::cout << "Norm of scaled residuals computation failed:\n";
          std::cout << std::scientific << std::setprecision(16)
                    << "\tMatrix inf  norm                 : " << inf_norm_A << "\n"
                    << "\tResidual inf norm                : " << inf_norm_res << "\n"
                    << "\tSolution inf norm                : " << inf_norm_x << "\n"
                    << "\tNorm of scaled residuals         : " << nsr_norm << "\n"
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

        real_type error = std::abs(norm_rhs_ * rrn_system - norm_res_) / norm_res_;
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
          std::cout << "Relative residual norm computation failed:\n";
          std::cout << std::scientific << std::setprecision(16)
                    << "\tTest value            : " << norm_res_ / norm_rhs_ << "\n"
                    << "\tSystemSolver computed : " << rrn_system << "\n\n";
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

        real_type error = std::abs(rn_system - norm_res_) / norm_res_;
        if (error > 10.0 * std::numeric_limits<real_type>::epsilon())
        {
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
        // Compute rhs and residual norms
        res_->copyDataFrom(r_, memspace_, memspace_);
        norm_rhs_ = norm2(*r_, memspace_);
        norm_res_ = computeResidualNorm(*A_, *x_, *res_, memspace_);

        // Compute norm of scaled residuals
        mh_.matrixInfNorm(A_, &inf_norm_A_, memspace_);
        inf_norm_x_   = vh_.infNorm(x_, memspace_);
        inf_norm_res_ = vh_.infNorm(res_, memspace_);
        nsr_norm_     = inf_norm_res_ / (inf_norm_A_ * inf_norm_x_);
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
      ReSolve::real_type computeResidualNorm(ReSolve::matrix::Sparse&     A,
                                             ReSolve::vector::Vector&     x,
                                             ReSolve::vector::Vector&     r,
                                             ReSolve::memory::MemorySpace memspace)
      {
        using namespace ReSolve::constants;
        mh_.matvec(&A, &x, &r, &ONE, &MINUS_ONE, memspace); // r := A * x - r
        return norm2(r, memspace);
      }

      /// Compute L2 norm of vector `r` in memory space `memspace`.
      ReSolve::real_type norm2(ReSolve::vector::Vector&     r,
                               ReSolve::memory::MemorySpace memspace)
      {
        return std::sqrt(vh_.dot(&r, &r, memspace));
      }

    private:
      ReSolve::matrix::Sparse* A_; ///< pointer to system matrix
      ReSolve::vector::Vector* r_; ///< pointer to system right-hand side
      ReSolve::vector::Vector* x_; ///< pointer to the computed solution

      ReSolve::MatrixHandler mh_; ///< matrix handler instance
      ReSolve::VectorHandler vh_; ///< vector handler instance

      ReSolve::vector::Vector* res_{nullptr};    ///< pointer to residual vector
      ReSolve::vector::Vector* x_true_{nullptr}; ///< pointer to solution error vector

      ReSolve::real_type norm_rhs_{0.0}; ///< right-hand side vector norm
      ReSolve::real_type norm_res_{0.0}; ///< residual vector norm

      real_type inf_norm_A_{0.0};   ///< infinity norm of matrix A
      real_type inf_norm_x_{0.0};   ///< infinity norm of solution x
      real_type inf_norm_res_{0.0}; ///< infinity norm of res = A*x - r
      real_type nsr_norm_{0.0};     ///< norm of scaled residuals

      ReSolve::memory::MemorySpace memspace_{ReSolve::memory::HOST};
      std::string                  hardware_backend_{"NONE"};
    };

  } // namespace examples
} // namespace ReSolve
