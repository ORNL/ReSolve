/**
 * @file testKLU_RocSolver.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@amd.com)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for rocsolver_rf.
 *
 */
#include <iomanip>
#include <iostream>
#include <string>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#endif

#include "TestHelper.hpp"

template <class workspace_type, class refactorization_type>
static int runTest(int argc, char* argv[], std::string& solver_name);

int main(int argc, char* argv[])
{
  using namespace ReSolve;
  int error_sum = 0;

#ifdef RESOLVE_USE_HIP
  std::string solver_name("rocsolverRf");
  error_sum += runTest<LinAlgWorkspaceHIP,
                       LinSolverDirectRocSolverRf>(argc, argv, solver_name);
#endif

#ifdef RESOLVE_USE_CUDA
  ReSolve::CliOptions options(argc, argv);

  auto        opt       = options.getParamFromKey("-s");
  std::string rf_solver = opt ? (*opt).second : "rf";
  if (rf_solver != "glu" && rf_solver != "rf")
  {
    std::cout << "Unrecognized refactorization solver " << rf_solver << " ...\n";
    std::cout << "Possible options are 'rf' and 'glu'.\n";
    std::cout << "Using default (rf) instead!\n";
    rf_solver = "rf";
  }
  if (rf_solver == "rf")
  {
    std::string solver_name("cusolverRf");
    error_sum += runTest<LinAlgWorkspaceCUDA,
                         LinSolverDirectCuSolverRf>(argc, argv, solver_name);
  }
  else
  {
    std::string solver_name("cusolverGLU");
    error_sum += runTest<LinAlgWorkspaceCUDA,
                         LinSolverDirectCuSolverGLU>(argc, argv, solver_name);
  }
#endif

  return error_sum;
}

template <class workspace_type, class refactorization_type>
int runTest(int argc, char* argv[], std::string& solver_name)
{
  std::string test_name("Test KLU with ");
  test_name += solver_name;

  // Use ReSolve data types.
  using index_type  = ReSolve::index_type;
  using real_type   = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using matrix_type = ReSolve::matrix::Sparse;

  // Error sum needs to be zero at the end for test to pass.
  int error_sum = 0;
  int status    = 0;

  // Collect all command line options
  ReSolve::CliOptions options(argc, argv);

  // Get directory with input files
  auto        opt       = options.getParamFromKey("-d");
  std::string data_path = opt ? (*opt).second : ".";

  // Change Rf solver mode (only for rocsolverRf)
  std::string mode("default");
  opt = options.getParamFromKey("-m");
  if (opt)
  {
    if (opt->second != "default" && opt->second != "rocsparse_trisolve")
    {
      std::cout << "Invalid rocSOLVER mode option.\n"
                << "Available modes are 'default' and 'rocsparse_trisolve'.\n"
                << "Setting mode to default ...\n";
    }
    else
    {
      mode = opt->second;
    }
  }

  // Whether to use iterative refinement
  opt        = options.getParamFromKey("-i");
  bool is_ir = opt ? true : false;

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create test helper
  TestHelper<workspace_type> helper(workspace);

  // Create direct solvers
  ReSolve::LinSolverDirectKLU KLU;
  refactorization_type        Rf(&workspace);
  if (solver_name == "rocsolverRf")
  {
    Rf.setCliParam("solve_mode", mode);
  }

  // Create iterative solver
  ReSolve::MatrixHandler            matrix_handler(&workspace);
  ReSolve::VectorHandler            vector_handler(&workspace);
  ReSolve::GramSchmidt              GS(&vector_handler, ReSolve::GramSchmidt::CGS2);
  ReSolve::LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Input data
  std::string matrix_file_name_1 = data_path + "/data/matrix_ACTIVSg200_AC_renumbered_add9_01.mtx";
  std::string matrix_file_name_2 = data_path + "/data/matrix_ACTIVSg200_AC_renumbered_add9_02.mtx";

  std::string rhs_file_name_1 = data_path + "/data/rhs_ACTIVSg200_AC_renumbered_add9_ones_01.mtx";
  std::string rhs_file_name_2 = data_path + "/data/rhs_ACTIVSg200_AC_renumbered_add9_ones_02.mtx";

  // Read first matrix
  std::ifstream mat1(matrix_file_name_1);
  if (!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  A->syncData(ReSolve::memory::DEVICE);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhs_file_name_1);
  if (!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name_1 << "\n";
    return -1;
  }
  real_type*  rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  vector_type vec_rhs(A->getNumRows());
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs.syncData(ReSolve::memory::DEVICE);
  rhs1_file.close();

  // Allocate the solution vector
  vector_type vec_x(A->getNumRows());
  vec_x.allocate(ReSolve::memory::HOST); // for KLU
  vec_x.allocate(ReSolve::memory::DEVICE);

  // Solve the first system using KLU
  status = KLU.setup(A);
  error_sum += status;

  status = KLU.analyze();
  error_sum += status;

  status = KLU.factorize();
  error_sum += status;

  // Extract factors and setup factorization for refactorization
  matrix_type* L = KLU.getLFactorCsr();
  matrix_type* U = KLU.getUFactorCsr();
  index_type*  P = KLU.getPOrdering();
  index_type*  Q = KLU.getQOrdering();

  status = Rf.setupCsr(A, L, U, P, Q, &vec_rhs);
  error_sum += status;

  // Refactorize (on device where available)
  status = Rf.refactorize();
  error_sum += status;

  // Solve system (on device where available)
  status = Rf.solve(&vec_rhs, &vec_x);
  error_sum += status;

  // Refine solutions
  if (is_ir)
  {
    test_name += " + IR";

    status = FGMRES.setup(A);
    error_sum += status;

    status = FGMRES.setupPreconditioner("LU", &Rf);
    error_sum += status;

    status = FGMRES.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }

  // Compute error norms for the system
  helper.setSystem(A, &vec_rhs, &vec_x);

  // Print result summary and check solution
  std::cout << "\nResults (first matrix): \n\n";
  helper.printSummary();
  if (is_ir)
  {
    helper.printIrSummary(&FGMRES);
  }
  error_sum += helper.checkResult(ReSolve::constants::MACHINE_EPSILON);

  // Load the second matrix
  std::ifstream mat2(matrix_file_name_2);
  if (!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  A->syncData(ReSolve::memory::DEVICE);
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
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  // Refactorize second matrix
  status = Rf.refactorize();
  error_sum += status;

  // Solve system (now one can go directly to IR when enabled)
  if (is_ir)
  {
    FGMRES.resetMatrix(A);
    status = FGMRES.setupPreconditioner("LU", &Rf);
    error_sum += status;

    status = FGMRES.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }
  else
  {
    status = Rf.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }

  // Recompute error norms for the second system and print summary
  helper.resetSystem(A, &vec_rhs, &vec_x);

  std::cout << "\nResults (second matrix): \n\n";
  helper.printSummary();
  if (is_ir)
  {
    helper.printIrSummary(&FGMRES);
  }
  error_sum += helper.checkResult(ReSolve::constants::MACHINE_EPSILON);

  isTestPass(error_sum, test_name);

  // delete data on the heap
  delete A;
  delete[] rhs;

  return error_sum;
}
