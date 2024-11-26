#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include "resolve/Common.hpp"
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>
#if defined RESOLVE_USE_KLU
#include <resolve/LinSolverDirectKLU.hpp>
#endif
#if defined (RESOLVE_USE_CUDA)
#include <resolve/LinSolverDirectCuSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
  std::string memory_space("cuda");
#elif defined (RESOLVE_USE_HIP)
#include <resolve/LinSolverDirectRocSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceHIP;
  std::string memory_space("hip");
#else
  using workspace_type = ReSolve::LinAlgWorkspaceCpu;
  std::string memory_space("cpu");
#endif
#include "FunctionalityTestHelper.hpp"

namespace ReSolve
{

  namespace tests
  {

    AxEqualsRhsProblem::~AxEqualsRhsProblem()
    {
      delete A_;
      delete vec_x_;
      delete vec_rhs_;
    }

    matrix::Csr* AxEqualsRhsProblem::getMatrix() const
    {
      return A_;
    }

    vector::Vector* AxEqualsRhsProblem::getVector() const
    {
      return vec_x_;
    }

    vector::Vector* AxEqualsRhsProblem::getRhs() const
    {
      return vec_rhs_;
    }

    void AxEqualsRhsProblem::updateProblem(const std::string& matrix_filepath, 
                                           const std::string& rhs_filepath)
    {
      // Load the second matrix
      std::ifstream mat2(matrix_filepath);
      if(!mat2.is_open()) {
        std::cout << "Failed to open file " << matrix_filepath << "\n";
        std::exit( 1 );
      }

      io::updateMatrixFromFile(mat2, A_);

      A_->syncData(memory::DEVICE);

      mat2.close();

      // Load the second rhs vector
      std::ifstream rhs2_file(rhs_filepath);
      if(!rhs2_file.is_open()) {
        std::cout << "Failed to open file " << rhs_filepath << "\n";
        std::exit( 1 );
      }

      // note: This intermediate allocation is clunky and should not be here
      // in the original test, it was updateArrayFromFile: but we should not have to
      // hold onto the intermediate rhs since it is just a temporary construction artifact
      real_type* rhs = io::createArrayFromFile(rhs2_file);
      //io::updateArrayFromFile(rhs2_file, &rhs);

      rhs2_file.close();

      vec_rhs_->update(rhs, memory::HOST, memory::DEVICE);

      delete[] rhs;
    }

    AxEqualsRhsProblem::AxEqualsRhsProblem(const std::string& matrix_filepath, 
                                           const std::string& rhs_filepath)
    {
      // Read first matrix
      std::ifstream mat1(matrix_filepath);
      if(!mat1.is_open()) {
        std::cout << "Failed to open file " << matrix_filepath << "\n";
        std::exit(1);
      }

      A_ = io::createCsrFromFile(mat1, true);
      A_->syncData(memory::DEVICE);
      mat1.close();

      // Read first rhs vector
      std::ifstream rhs1_file(rhs_filepath);
      if(!rhs1_file.is_open()) {
        std::cout << "Failed to open file " << rhs_filepath << "\n";
        std::exit(1);
      }
      real_type* rhs = io::createArrayFromFile(rhs1_file);
      rhs1_file.close();

      // setup/allocate testing workspace phase:

      // Create rhs, solution and residual vectors
      vec_rhs_ = new vector::Vector(A_->getNumRows());
      vec_x_   = new vector::Vector(A_->getNumRows());

      // Allocate solution vector
      vec_x_->allocate(memory::HOST);  //for KLU
      vec_x_->allocate(memory::DEVICE);

      // Set RHS vector on CPU (update function allocates)
      vec_rhs_->update(rhs, memory::HOST, memory::HOST);

      delete[] rhs;
    }

    real_type FunctionalityTestHelper::calculateSolutionVectorNorm(vector::Vector& vec_x)
    {
      using namespace memory;

      // set member variable and also return in case this function is used outside of this class
      norm_solution_vector_ = sqrt(vh_->dot(&vec_x, &vec_x, DEVICE));

      return norm_solution_vector_;
    }

    real_type FunctionalityTestHelper::calculateRhsVectorNorm(vector::Vector& vec_rhs)
    {
      using namespace memory;

      // set member variable and also return in case this function is used outside of this class
      rhs_norm_ = sqrt(vh_->dot(&vec_rhs, &vec_rhs, DEVICE));

      return rhs_norm_;
    }

    real_type FunctionalityTestHelper::calculateResidualNorm(vector::Vector& vec_r)
    {
      using namespace memory;

      // set member variable and also return in case this function is used outside of this class
      residual_norm_ = sqrt(vh_->dot(&vec_r, &vec_r, DEVICE));

      return residual_norm_;
    }

    real_type FunctionalityTestHelper::calculateDiffNorm(vector::Vector& vec_x)
    {
      using namespace memory;
      using namespace ReSolve::constants;

      index_type n = vec_x.getSize();

      vector::Vector vec_diff(n);

      vec_diff.setToConst(1.0, DEVICE);

      // why does this not return an error status if it fails?
      vh_->axpy(&MINUSONE, &vec_x, &vec_diff, DEVICE);

      diff_norm_ = sqrt(vh_->dot(&vec_diff, &vec_diff, DEVICE));

      return diff_norm_;
    }

    real_type FunctionalityTestHelper::calculateTrueNorm(matrix::Csr& A, vector::Vector& vec_rhs)
    {
      using namespace memory;
      using namespace ReSolve::constants;

      index_type n = A.getNumRows();

      vector::Vector vec_test(n);
      vector::Vector vec_tmp(n);

      //compute the residual using exact solution
      vec_test.setToConst(1.0, DEVICE);
      vec_tmp.update(vec_rhs.getData(HOST), HOST, DEVICE);
      int status = mh_->matvec(&A, &vec_test, &vec_tmp, &ONE, &MINUSONE, DEVICE); 

      if (status != 0) {
        std::cout << "matvec failed" << std::endl;
        std::exit( status );
      }

      true_residual_norm_ = sqrt(vh_->dot(&vec_tmp, &vec_tmp, DEVICE));

      return true_residual_norm_;
    }

    vector::Vector FunctionalityTestHelper::generateResidualVector(matrix::Csr& A,
                                                                   vector::Vector& vec_x,
                                                                   vector::Vector& vec_rhs)
    {
      using namespace memory;
      using namespace ReSolve::constants;

      index_type n = A.getNumRows();

      vector::Vector vec_r(n);

      vec_r.update(vec_rhs.getData(HOST), HOST, DEVICE);

      mh_->setValuesChanged(true, DEVICE);

      int status = mh_->matvec(&A, &vec_x, &vec_r, &ONE, &MINUSONE, DEVICE); 

      if (status != 0) {
        std::cout << "matvec from matrixhandler failed" << std::endl;

        std::exit(status);
      }

      return vec_r;
    }

    void FunctionalityTestHelper::printNorms(std::string &testname)
    {
      std::cout << "Results for " << testname << ":\n\n";

      std::cout << std::scientific << std::setprecision(16);

      std::cout << "\t ||b-A*x||_2                 : " 
                << residual_norm_           
                << " (residual norm)\n";

      std::cout << "\t ||b-A*x||_2/||b||_2         : " 
                << residual_norm_/rhs_norm_ 
                << " (relative residual norm)\n";

      std::cout << "\t ||x-x_true||_2              : " 
                << diff_norm_               
                << " (solution error)\n";

      std::cout << "\t ||x-x_true||_2/||x_true||_2 : " 
                << diff_norm_/norm_solution_vector_    
                << " (relative solution error)\n";

      std::cout << "\t ||b-A*x_exact||_2           : " 
                << true_residual_norm_      
                << " (control; residual norm with exact solution)\n";
    }

    int FunctionalityTestHelper::checkResidualNorm()
    {
      int error_sum = 0;

      if (!std::isfinite(residual_norm_/rhs_norm_)) {
        std::cout << "Result is not a finite number!\n";
        error_sum++;
      }

      if (residual_norm_/rhs_norm_ > tol_) {
        std::cout << "Result inaccurate!\n";
        error_sum++;
      }

      return error_sum;
    }

    void FunctionalityTestHelper::calculateNorms( AxEqualsRhsProblem &problem )
    {
      using namespace memory;
      using namespace ReSolve::constants;

      matrix::Csr& A = *problem.getMatrix();
      vector::Vector& vec_x = *problem.getVector();
      vector::Vector& vec_rhs = *problem.getRhs();

      int error_sum = 0;

      calculateSolutionVectorNorm(vec_x);

      // Compute residual norm
      vector::Vector vec_r = generateResidualVector(A, vec_x, vec_rhs);

      calculateResidualNorm(vec_r);

      //for testing only - control
      calculateRhsVectorNorm(vec_rhs);

      //compute ||x_diff|| = ||x - x_true|| norm
      calculateDiffNorm(vec_x);

      calculateTrueNorm(A, vec_rhs);
    }

    // note: this should be split into two functions for separate printing and checking
    int FunctionalityTestHelper::checkResult(matrix::Csr& A,
                                             vector::Vector& vec_rhs,
                                             vector::Vector& vec_x,
                                             SystemSolver& solver,
                                             std::string testname)
    {
      using namespace memory;
      using namespace ReSolve::constants;

      int error_sum = 0;

      error_sum += checkResidualNorm();

      printNorms(testname);
      
      printIterativeSolverStats(solver);

      return error_sum;
    }

    FunctionalityTestHelper::FunctionalityTestHelper(real_type tol_init,
                                                     workspace_type &workspace_init,
                                                     AxEqualsRhsProblem &problem)
      : tol_(tol_init),
        workspace_(workspace_init)
    {
      //mh_ = new MatrixHandler(&workspace_);
      mh_ = std::make_unique<ReSolve::MatrixHandler>(&workspace_);

      vh_ = std::make_unique<ReSolve::VectorHandler>(&workspace_);

      calculateNorms(problem);
    }

    FunctionalityTestHelper::~FunctionalityTestHelper()
    {
      // no longer needed if mh_ is a unique_ptr, it will be deleted when out of scope
      //delete mh_;

      //delete vh_;
    }

    void FunctionalityTestHelper::printIterativeSolverStats(SystemSolver& solver)
    {
      // Get solver parameters
      real_type tol = solver.getIterativeSolver().getTol();
      index_type restart = solver.getIterativeSolver().getRestart();
      index_type maxit = solver.getIterativeSolver().getMaxit();

      // Get solver stats
      index_type num_iter   = solver.getIterativeSolver().getNumIter();
      real_type init_rnorm  = solver.getIterativeSolver().getInitResidualNorm();
      real_type final_rnorm = solver.getIterativeSolver().getFinalResidualNorm();
      
      std::cout << "\t IR iterations               : " << num_iter    << " (max " << maxit << ", restart " << restart << ")\n";
      std::cout << "\t IR starting res. norm       : " << init_rnorm  << "\n";
      std::cout << "\t IR final res. norm          : " << final_rnorm << " (tol " << std::setprecision(2) << tol << ")\n\n";
    }

    int FunctionalityTestHelper::checkNormOfScaledResiduals(matrix::Csr& A,
                                                            vector::Vector& vec_rhs,
                                                            vector::Vector& vec_x,
                                                            SystemSolver& solver)
    {
      using namespace ReSolve::constants;
      using namespace memory;
      int error_sum = 0;

      // Compute residual norm for the second system
      vector::Vector vec_r = generateResidualVector(A, vec_x, vec_rhs);

      // Compute norm of scaled residuals for the second system
      real_type inf_norm_A = 0.0;  
      mh_->matrixInfNorm(&A, &inf_norm_A, DEVICE); 
      real_type inf_norm_x = vh_->infNorm(&vec_x, DEVICE);
      real_type inf_norm_r = vh_->infNorm(&vec_r, DEVICE);
      real_type nsr_norm   = inf_norm_r / (inf_norm_A * inf_norm_x);
      real_type nsr_system = solver.getNormOfScaledResiduals(&vec_rhs, &vec_x);
      real_type error      = std::abs(nsr_system - nsr_norm)/nsr_norm;

      if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
        std::cout << "Norm of scaled residuals computation failed:\n";
        std::cout << std::scientific << std::setprecision(16)
                  << "\tMatrix inf  norm                 : " << inf_norm_A << "\n"
                  << "\tResidual inf norm                : " << inf_norm_r << "\n"  
                  << "\tSolution inf norm                : " << inf_norm_x << "\n"  
                  << "\tNorm of scaled residuals         : " << nsr_norm   << "\n"
                  << "\tNorm of scaled residuals (system): " << nsr_system << "\n\n";
        error_sum++;
      }

      return error_sum;
    }


    // note: pass rel_residual_norm in as a function parameter rather than calculate here
    int FunctionalityTestHelper::checkRelativeResidualNorm(vector::Vector& vec_rhs,
                                                           vector::Vector& vec_x,
                                                           SystemSolver& solver)
    {
      using namespace memory;

      int error_sum = 0;

      real_type rel_residual_norm = solver.getResidualNorm(&vec_rhs, &vec_x);

      real_type error = std::abs(rhs_norm_ * rel_residual_norm - residual_norm_)/residual_norm_;

      if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {

        std::cout << "Relative residual norm computation failed:\n";

        std::cout << std::scientific << std::setprecision(16)
          << "\tTest value            : " << residual_norm_/rhs_norm_ << "\n"
          << "\tSystemSolver computed : " << rel_residual_norm   << "\n\n";

        error_sum++;
      }

      return error_sum;
    }

  } //namespace tests
} //namespace ReSolve
