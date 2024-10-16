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
