/**
 * @file testSysHipRefine.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for SystemSolver class
 * @date 2023-12-14
 * 
 * 
 */
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

#include <tests/functionality/FunctionalityTestHelper.hpp>


using namespace ReSolve::constants;
using namespace ReSolve::tests;
using namespace ReSolve::colors;


int main(int argc, char *argv[])
{
  using real_type   = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  // rhs for rhs (solution b vector) ax = b, and first part is model (Texas grid)
  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg2000_AC_00.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg2000_AC_02.mtx";

  // so we are setting up A_above * x_unknown = rhs_below, 
  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg2000_AC_00.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg2000_AC_02.mtx.ones";

  // axb problem construction
  AxEqualsRhsProblem axb(matrixFileName1, rhsFileName1);
  ReSolve::matrix::Csr* A = axb.getMatrix();
  vector_type* vec_x   = axb.getVector();
  vector_type* vec_rhs = axb.getRhs();

  // Error sum needs to be 0 at the end for test to PASS.
  // It is a FAIL otheriwse.
  int error_sum = 0;
  int status = 0;

  // workspace construction
  workspace_type workspace;

  workspace.initializeHandles();

  // system solver construction uses workspace
  ReSolve::SystemSolver solver(&workspace);

  // Configure solver (CUDA-based solver needs slightly different
  // settings than HIP-based one)
  // cgs2 = classical Gram-Schmidt
  solver.setRefinementMethod("fgmres", "cgs2");

  solver.getIterativeSolver().setRestart(100);

  if (memory_space == "hip") {
    solver.getIterativeSolver().setMaxit(200);
  }

  if (memory_space == "cuda") {
    solver.getIterativeSolver().setMaxit(400);
    solver.getIterativeSolver().setTol(1e-17);
  }

  // end solver creation

  // next connect solver to problem:

  status = solver.setMatrix(A);
  error_sum += status;

  // Solve the first system using KLU
  status = solver.analyze();
  error_sum += status;

  status = solver.factorize();
  error_sum += status;

  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;

  // construct test helper object

  // larger tolerance than default 1e-17 because iterative refinement is not applied here
  ReSolve::tests::FunctionalityTestHelper test_helper(1e-12, workspace);

  // the below step should happen in a constructor, as it is extremely easy to forget...
  // it should then be re-constructed a second time. No more memory allocations should happen...
  // Q: is the norm calculation functionality the same in all the tests?
  test_helper.calculateNorms(axb);

  // Verify relative residual norm computation in SystemSolver
  error_sum += test_helper.checkRelativeResidualNorm(*vec_rhs, *vec_x, solver);

  // Compute norm of scaled residuals
  error_sum += test_helper.checkNormOfScaledResiduals(*A, *vec_rhs, *vec_x, solver);

  error_sum += 
  test_helper.checkResult(*A, *vec_rhs, *vec_x, solver, "first matrix");

  // Now prepare the Rf solver
  status = solver.refactorizationSetup();
  error_sum += status;

  axb.updateProblem(matrixFileName2, rhsFileName2);

  status = solver.refactorize();

  error_sum += status;
  
  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;

  test_helper.calculateNorms(axb);

  // Verify relative residual norm computation in SystemSolver
  error_sum += test_helper.checkRelativeResidualNorm(*vec_rhs, *vec_x, solver);

  // Compute norm of scaled residuals
  error_sum += test_helper.checkNormOfScaledResiduals(*A, *vec_rhs, *vec_x, solver);

  error_sum += 
  test_helper.checkResult(*A, *vec_rhs, *vec_x, solver, "second matrix");

  if (error_sum == 0) {
    std::cout << "Test KLU with Rf solver + IR " << GREEN << "PASSED" << CLEAR <<std::endl<<std::endl;;
  } else {
    std::cout << "Test KLU with Rf solver + IR " << RED << "FAILED" << CLEAR << ", error sum: "<<error_sum<<std::endl<<std::endl;;
  }

  // if not zero, main() exits with problems
  return error_sum;
}
