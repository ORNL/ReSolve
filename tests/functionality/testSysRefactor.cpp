
/**
 * @file testSysHipRefine.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for SystemSolver class
 * @date 2023-12-14
 *
 *
 */
#include <iomanip>
#include <iostream>
#include <string>

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif

#include "TestHelper.hpp"

template <class workspace_type>
static int runTest(int argc, char* argv[], std::string backend);

int main(int argc, char* argv[])
{
  int error_sum = 0;

  // Refactorization on CPU not currently supported in SystemSolver class
  // error_sum += runTest<ReSolve::LinAlgWorkspaceCpu>(argc, argv, "cpu");

#ifdef RESOLVE_USE_CUDA
  error_sum += runTest<ReSolve::LinAlgWorkspaceCUDA>(argc, argv, "cuda");
#endif

#ifdef RESOLVE_USE_HIP
  error_sum += runTest<ReSolve::LinAlgWorkspaceHIP>(argc, argv, "hip");
#endif

  return error_sum;
}

template <class workspace_type>
static int runTest(int argc, char* argv[], std::string backend)
{
  // Use ReSolve data types.
  using namespace ReSolve;
  using real_type   = ReSolve::real_type;
  using index_type  = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;

  // Error sum needs to be 0 at the end for test to PASS.
  // It is a FAIL otheriwse.
  int error_sum = 0;
  int status    = 0;

  memory::MemorySpace memspace = memory::HOST;
  if (backend != "cpu")
  {
    memspace = memory::DEVICE;
  }

  // Collect all command line options
  ReSolve::CliOptions options(argc, argv);

  // Get directory with input files
  auto        opt       = options.getParamFromKey("-d");
  std::string data_path = opt ? (*opt).second : ".";

  // Get matrix file name
  opt = options.getParamFromKey("-m");
  if (!opt)
  {
    std::cout << "Matrix file name not provided. Use -m <matrix_file_name>.\n";
    return -1;
  }
  std::string matrix_temp = (*opt).second;

  // Get rhs file name
  opt = options.getParamFromKey("-r");
  if (!opt)
  {
    std::cout << "RHS file name not provided. Use -r <rhs_file_name>.\n";
    return -1;
  }
  std::string rhs_temp = (*opt).second;

  // Construct matrix and rhs file names from inputs
  std::string matrix_file_name_1 = data_path + matrix_temp + "01.mtx";
  std::string matrix_file_name_2 = data_path + matrix_temp + "02.mtx";
  std::string rhs_file_name_1    = data_path + rhs_temp + "01.mtx";
  std::string rhs_file_name_2    = data_path + rhs_temp + "02.mtx";

  // Create workspace and initialize its handles.
  workspace_type workspace;
  workspace.initializeHandles();

  // Create test helper
  TestHelper<workspace_type> helper(workspace);

  // Create system solver
  std::string refactor("none");
  if (backend == "cuda")
  {
    refactor = "cusolverrf";
  }
  else if (backend == "hip")
  {
    refactor = "rocsolverrf";
  }
  else
  {
    refactor = "klu";
  }
  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               refactor, // refactorization
                               refactor, // triangular solve
                               "none",   // preconditioner (always 'none' here)
                               "none");   // iterative refinement

  // Configure solver (CUDA-based solver needs slightly different
  // settings than HIP-based one)
  solver.setRefinementMethod("fgmres", "cgs2");
  solver.getIterativeSolver().setCliParam("restart", "100");
  solver.getIterativeSolver().setTol(ReSolve::constants::MACHINE_EPSILON);
  if (backend == "hip")
  {
    solver.getIterativeSolver().setMaxit(200);
  }
  if (backend == "cuda")
  {
    solver.getIterativeSolver().setMaxit(400);
  }

  // Read first matrix
  std::ifstream mat1(matrix_file_name_1);
  if (!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1, true);
  if (memspace != memory::HOST)
  {
    A->syncData(memspace);
  }
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhs_file_name_1);
  if (!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name_1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  rhs1_file.close();

  // Create and set residual vector
  vector_type vec_rhs = (A->getNumRows());
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  // Create and allocate solution vector
  vector_type vec_x(A->getNumRows());
  if (memspace != memory::HOST)
  {
    vec_x.allocate(ReSolve::memory::HOST); // for KLU
  }
  vec_x.allocate(memspace);

  // Add system matrix to the solver
  status = solver.setMatrix(A);
  error_sum += status;

  // Solve the first system using KLU
  status = solver.analyze();
  error_sum += status;

  status = solver.factorize();
  error_sum += status;

  status = solver.solve(&vec_rhs, &vec_x);
  error_sum += status;

  // Compute error norms for the system
  helper.setSystem(A, &vec_rhs, &vec_x);

  // Print result summary and check solution
  std::cout << "\nResults (first matrix): \n\n";
  helper.printSummary();
  error_sum += helper.checkResult(ReSolve::constants::MACHINE_EPSILON);

  // Verify norm of scaled residuals calculation in SystemSolver class
  real_type nsr_system = solver.getNormOfScaledResiduals(&vec_rhs, &vec_x);
  error_sum += helper.checkNormOfScaledResiduals(nsr_system);

  // Verify relative residual norm computation in SystemSolver
  real_type rel_residual_norm = solver.getResidualNorm(&vec_rhs, &vec_x);
  error_sum += helper.checkRelativeResidualNorm(rel_residual_norm);

  // Now prepare the Rf solver
  status = solver.refactorizationSetup();
  error_sum += status;

  // Load the second matrix
  std::ifstream mat2(matrix_file_name_2);
  if (!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  if (memspace != memory::HOST)
  {
    A->syncData(memspace);
  }
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhs_file_name_2);
  if (!rhs2_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name_2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();

  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, memspace);

  // Refactorize matrix
  status = solver.refactorize();
  error_sum += status;

  // Solve system
  status = solver.solve(&vec_rhs, &vec_x);
  error_sum += status;

  // Compute error norms for the system
  helper.resetSystem(A, &vec_rhs, &vec_x);

  // Print result summary and check solution
  std::cout << "\nResults (second matrix): \n\n";
  helper.printSummary();
  helper.printIrSummary(&(solver.getIterativeSolver()));
  error_sum += helper.checkResult(ReSolve::constants::MACHINE_EPSILON);

  // Verify norm of scaled residuals calculation in SystemSolver class
  nsr_system = solver.getNormOfScaledResiduals(&vec_rhs, &vec_x);
  error_sum += helper.checkNormOfScaledResiduals(nsr_system);

  // Verify relative residual norm computation in SystemSolver
  rel_residual_norm = solver.getResidualNorm(&vec_rhs, &vec_x);
  error_sum += helper.checkRelativeResidualNorm(rel_residual_norm);

  // Add one output specific to GMRES
  index_type restart = solver.getIterativeSolver().getCliParamInt("restart");
  std::cout << "\t IR GMRES restart                                : " << restart << "\n";

  isTestPass(error_sum, "Test SystemSolver: KLU with Rf and IR");

  delete A;
  delete[] rhs;

  return error_sum;
}
