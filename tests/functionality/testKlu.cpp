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

#include "TestHelper.hpp"
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

template <class workspace_type, class refactorization_type>
static int runTest(int argc, char* argv[], std::string& solver_name);

int main(int argc, char* argv[])
{
  using namespace ReSolve;
  int error_sum = 0;

  std::string solver_name("KLU refactorization");
  error_sum += runTest<LinAlgWorkspaceCpu,
                       LinSolverDirectKLU>(argc, argv, solver_name);

  return error_sum;
}

template <class workspace_type, class refactorization_type>
int runTest(int argc, char* argv[], std::string& solver_name)
{
  std::string test_name("Test KLU with ");
  test_name += solver_name;

  // Use ReSolve data types.
  using real_type   = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  // Error sum needs to be zero at the end for test to pass.
  int error_sum = 0;
  int status    = 0;

  // Collect all command line options
  ReSolve::CliOptions options(argc, argv);

  // Get directory with input files
  auto        opt       = options.getParamFromKey("-d");
  std::string data_path = opt ? (*opt).second : ".";

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

  // Create iterative solver
  ReSolve::MatrixHandler            matrix_handler(&workspace);
  ReSolve::VectorHandler            vector_handler(&workspace);
  ReSolve::GramSchmidt              GS(&vector_handler, ReSolve::GramSchmidt::CGS2);
  ReSolve::LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Input data
  std::string matrix_file_name_1 = data_path + "/data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrix_file_name_2 = data_path + "/data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhs_file_name_1 = data_path + "/data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhs_file_name_2 = data_path + "/data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix
  std::ifstream mat1(matrix_file_name_1);
  if (!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
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
  rhs1_file.close();

  // Allocate the solution vector
  vector_type vec_x(A->getNumRows());
  vec_x.allocate(ReSolve::memory::HOST); // for KLU

  // Solve the first system using KLU
  status = KLU.setup(A);
  error_sum += status;

  status = KLU.analyze();
  error_sum += status;

  status = KLU.factorize();
  error_sum += status;

  std::cout << "KLU factorize status: " << status << std::endl;

  status = KLU.solve(&vec_rhs, &vec_x);
  error_sum += status;

  if (is_ir)
  {
    test_name += " + IR";

    status = FGMRES.setup(A);
    error_sum += status;

    status = FGMRES.setupPreconditioner("LU", &KLU);
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
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  status = KLU.refactorize();
  error_sum += status;
  std::cout << "KLU refactorization status: " << status << std::endl;

  if (is_ir)
  {
    FGMRES.resetMatrix(A);
    status = FGMRES.setupPreconditioner("LU", &KLU);
    error_sum += status;

    status = FGMRES.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }
  else
  {
    status = KLU.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }

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
