#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

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

real_type FunctionalityTestHelper::
calculateSolutionVectorNorm(ReSolve::vector::Vector& vec_x)
{
  using namespace memory;

  // set member variable and also return in case this function is used outside of this class
  norm_solution_vector_ = sqrt(vh_->dot(&vec_x, &vec_x, DEVICE));

  return norm_solution_vector_;
}

real_type FunctionalityTestHelper::
calculate_rhs_vector_norm(ReSolve::vector::Vector& vec_rhs)
{
  using namespace memory;

  // set member variable and also return in case this function is used outside of this class
  rhs_norm_ = sqrt(vh_->dot(&vec_rhs, &vec_rhs, DEVICE));

  return rhs_norm_;
}

real_type FunctionalityTestHelper::
calculateResidualNorm(ReSolve::vector::Vector& vec_r)
{
  using namespace memory;

  // set member variable and also return in case this function is used outside of this class
  residual_norm_ = sqrt(vh_->dot(&vec_r, &vec_r, DEVICE));

  return residual_norm_;
}

real_type FunctionalityTestHelper::
calculate_diff_norm(ReSolve::matrix::Csr& A, ReSolve::vector::Vector& vec_x)
{
  using namespace memory;
  using namespace ReSolve::constants;

  index_type n = A.getNumRows();

  ReSolve::vector::Vector vec_diff(n);

  vec_diff.setToConst(1.0, DEVICE);

  // why does this not return an error status if it fails?
  vh_->axpy(&MINUSONE, &vec_x, &vec_diff, DEVICE);

  diff_norm_ = sqrt(vh_->dot(&vec_diff, &vec_diff, DEVICE));

  return diff_norm_;
}

real_type FunctionalityTestHelper::
calculate_true_norm(ReSolve::matrix::Csr& A, ReSolve::vector::Vector& vec_rhs)
{
  using namespace memory;
  using namespace ReSolve::constants;

  index_type n = A.getNumRows();

  ReSolve::vector::Vector vec_test(n);
  ReSolve::vector::Vector vec_tmp(n);

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

ReSolve::vector::Vector FunctionalityTestHelper::
generateResidualVector(ReSolve::matrix::Csr& A,
                         ReSolve::vector::Vector& vec_x,
                         ReSolve::vector::Vector& vec_rhs)
{
  using namespace memory;
  using namespace ReSolve::constants;

  index_type n = A.getNumRows();

  ReSolve::vector::Vector vec_r(n);

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

void FunctionalityTestHelper::calculateNorms(
    ReSolve::matrix::Csr& A,
    ReSolve::vector::Vector& vec_rhs,
    ReSolve::vector::Vector& vec_x )
{
  using namespace memory;
  using namespace ReSolve::constants;

  int error_sum = 0;

  calculateSolutionVectorNorm(vec_x);

  // Compute residual norm
  ReSolve::vector::Vector vec_r = generateResidualVector(A, vec_x, vec_rhs);

  calculateResidualNorm(vec_r);

  //for testing only - control
  calculate_rhs_vector_norm(vec_rhs);

  //compute ||x_diff|| = ||x - x_true|| norm
  calculate_diff_norm(A, vec_x);

  calculate_true_norm(A, vec_rhs);
}

int FunctionalityTestHelper::checkResult(
    ReSolve::matrix::Csr& A,
    ReSolve::vector::Vector& vec_rhs,
    ReSolve::vector::Vector& vec_x,
    ReSolve::SystemSolver& solver,
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

FunctionalityTestHelper::FunctionalityTestHelper( 
  ReSolve::real_type tol_init)
  :
  tol_(tol_init)
{
  workspace_.initializeHandles();

  mh_ = new ReSolve::MatrixHandler(&workspace_);

  vh_ = new ReSolve::VectorHandler(&workspace_);
}

FunctionalityTestHelper::~FunctionalityTestHelper()
{
  delete mh_;

  delete vh_;
}

void FunctionalityTestHelper::printIterativeSolverStats(SystemSolver& solver)
{
  // Get solver parameters
  real_type tol = solver.getIterativeSolver().getTol();
  index_type restart = solver.getIterativeSolver().getRestart();
  index_type maxit = solver.getIterativeSolver().getMaxit();

  // note: these are the solver's tolerance, different from the testhelper's tolerance

  // Get solver stats
  index_type num_iter   = solver.getIterativeSolver().getNumIter();
  real_type init_rnorm  = solver.getIterativeSolver().getInitResidualNorm();
  real_type final_rnorm = solver.getIterativeSolver().getFinalResidualNorm();
  
  std::cout << "\t IR iterations               : " << num_iter    << " (max " << maxit << ", restart " << restart << ")\n";
  std::cout << "\t IR starting res. norm       : " << init_rnorm  << "\n";
  std::cout << "\t IR final res. norm          : " << final_rnorm << " (tol " << std::setprecision(2) << tol << ")\n\n";
}

int FunctionalityTestHelper::checkNormOfScaledResiduals(ReSolve::matrix::Csr& A,
                              ReSolve::vector::Vector& vec_rhs,
                              ReSolve::vector::Vector& vec_x,
                              ReSolve::SystemSolver& solver)
{
  using namespace ReSolve::constants;
  using namespace memory;
  int error_sum = 0;

  // Compute residual norm for the second system
  ReSolve::vector::Vector vec_r = generateResidualVector( A, vec_x, vec_rhs );

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


int FunctionalityTestHelper::checkRelativeResidualNorm(ReSolve::vector::Vector& vec_rhs,
    ReSolve::vector::Vector& vec_x,
    ReSolve::SystemSolver& solver)
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
