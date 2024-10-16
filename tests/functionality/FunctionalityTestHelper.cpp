#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>

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

namespace ReSolve {
  namespace tests{

FunctionalityTestHelper::FunctionalityTestHelper( 
  ReSolve::real_type tol_init )
  :
  tol_( tol_init )
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

int FunctionalityTestHelper::checkNormOfScaledResiduals(ReSolve::matrix::Csr& A,
                              ReSolve::vector::Vector& vec_rhs,
                              ReSolve::vector::Vector& vec_x,
                              ReSolve::vector::Vector& vec_r,
                              ReSolve::SystemSolver& solver)
{
  using namespace ReSolve::constants;
  using namespace memory;
  int error_sum = 0;

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
    const real_type residual_norm,
    const real_type rhs_norm,
    ReSolve::SystemSolver& solver)
{
  using namespace memory;
  int error_sum = 0;

  real_type rel_residual_norm = solver.getResidualNorm(&vec_rhs, &vec_x);
  real_type error = std::abs(rhs_norm * rel_residual_norm - residual_norm)/residual_norm;
  if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
    std::cout << "Relative residual norm computation failed:\n";
    std::cout << std::scientific << std::setprecision(16)
      << "\tTest value            : " << residual_norm/rhs_norm << "\n"
      << "\tSystemSolver computed : " << rel_residual_norm   << "\n\n";
    error_sum++;
  }

  return error_sum;
}

}
}
