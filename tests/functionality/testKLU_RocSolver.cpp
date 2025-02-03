/**
 * @file testKLU_RocSolver.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@amd.com)
 * @brief Functionality test for rocsolver_rf.
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/utilities/params/CliOptions.hpp>

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#endif

#include "TestHelper.hpp"

template <class workspace_type, class solver_type>
static int runTest(int argc, char *argv[], std::string& solver_name);

int main(int argc, char *argv[])
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
  ReSolve::CliOptions::Option* opt = nullptr;

  opt = options.getParamFromKey("-s");
  std::string rf_solver = opt ? (*opt).second : "rf";
  if (rf_solver != "glu" && rf_solver != "rf") {
    std::cout << "Unrecognized refactorization solver " << rf_solver << " ...\n";
    std::cout << "Possible options are 'rf' and 'glu'.\n";
    std::cout << "Using default (rf) instead!\n";
    rf_solver = "rf";
  }
  if (rf_solver == "rf") {
    std::string solver_name("cusolverRf");
    error_sum += runTest<LinAlgWorkspaceCUDA,
                         LinSolverDirectCuSolverRf>(argc, argv, solver_name);
  } else {
    std::string solver_name("cusolverGLU");
    error_sum += runTest<LinAlgWorkspaceCUDA,
                         LinSolverDirectCuSolverGLU>(argc, argv, solver_name);
  }
#endif

  return error_sum;
}

template <class workspace_type, class solver_type>
int runTest(int argc, char *argv[], std::string& solver_name)
{
  std::string test_name("Test KLU with ");
  test_name += solver_name;

  // Use ReSolve data types.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using matrix_type = ReSolve::matrix::Sparse;

  // Error sum needs to be zero at the end for test to pass.
  int error_sum = 0;
  int status = 0;

  // Collect all CLI
  ReSolve::CliOptions options(argc, argv);
  ReSolve::CliOptions::Option* opt = nullptr;

  // Get directory with input files
  opt = options.getParamFromKey("-d");
  std::string data_path = opt ? (*opt).second : "./";

  // Change Rf solver mode (only for rocsolverRf)
  // Mode 1 uses rocSPARSE triangular solver instead of rocSOLVER default
  int mode = 0;
  opt = options.getParamFromKey("-m");
  if (opt) {
    mode = 1;
    test_name += " (mode 1)";
  }

  // Whether to use iterative refinement
  opt = options.getParamFromKey("-i");
  bool is_ir = opt ? true : false;

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create direct solvers
  ReSolve::LinSolverDirectKLU KLU;
  solver_type Rf(&workspace);
  // Rf.setSolveMode(mode);

  // Create iterative solver
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);
  ReSolve::GramSchmidt GS(&vector_handler, ReSolve::GramSchmidt::CGS2);
  ReSolve::LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);
  FGMRES.setMaxit(200); 
  FGMRES.setRestart(100); 


  std::string matrix_file_name_1 = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrix_file_name_2 = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhs_file_name_1 = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhs_file_name_2 = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix
  std::ifstream mat1(matrix_file_name_1);
  if(!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  A->syncData(ReSolve::memory::DEVICE);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhs_file_name_1);
  if(!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name_1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  vector_type vec_rhs(A->getNumRows());
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs.syncData(ReSolve::memory::DEVICE);
  rhs1_file.close();

  // Allocate the solution vector
  vector_type vec_x(A->getNumRows());
  vec_x.allocate(ReSolve::memory::HOST); //for KLU
  vec_x.allocate(ReSolve::memory::DEVICE);

  // Solve the first system using KLU
  status = KLU.setup(A);
  error_sum += status;

  status = KLU.analyze();
  error_sum += status;

  status = KLU.factorize();
  error_sum += status;

  std::cout << "KLU factorize status: " << status <<std::endl;      

  // status = KLU.solve(&vec_rhs, &vec_x);
  // error_sum += status;

  matrix_type* L = KLU.getLFactor();
  matrix_type* U = KLU.getUFactor();
  index_type* P = KLU.getPOrdering();
  index_type* Q = KLU.getQOrdering();

  status = Rf.setup(A, L, U, P, Q, &vec_rhs); 
  error_sum += status;
  std::cout << "Rf setup status: " << status << std::endl;      

  status = Rf.refactorize();
  error_sum += status;
  std::cout << "Rf refactorize status: " << status << std::endl;      

  status = Rf.solve(&vec_rhs, &vec_x);
  error_sum += status;
  std::cout << "Rf solve status: " << status << std::endl;      

  if (is_ir) {
    test_name += " + IR";

    status =  FGMRES.setup(A); 
    error_sum += status;

    status = FGMRES.setupPreconditioner("LU", &Rf);
    error_sum += status;

    status = FGMRES.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }

  // Setup test helper
  TestHelper<workspace_type> th(A, &vec_rhs, &vec_x, workspace);

  // Print result summary and check solution
  std::cout << "\nResults (first matrix): \n\n";
  th.printSummary();
  if (is_ir) {
    th.printIrSummary(&FGMRES);
  }
  error_sum += th.checkResult(1e-16);

  // Load the second matrix
  std::ifstream mat2(matrix_file_name_2);
  if(!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name_2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  A->syncData(ReSolve::memory::DEVICE);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhs_file_name_2);
  if(!rhs2_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name_2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  status = Rf.refactorize();
  error_sum += status;
  std::cout << "rocSolverRf refactorization status: " << status << std::endl;      
  
  if (is_ir) {
    FGMRES.resetMatrix(A);
    status = FGMRES.setupPreconditioner("LU", &Rf);
    error_sum += status;

    status = FGMRES.solve(&vec_rhs, &vec_x);
    error_sum += status;
  } else {
    status = Rf.solve(&vec_rhs, &vec_x);
    error_sum += status;
  }

  th.resetSystem(A, &vec_rhs, &vec_x);

  std::cout << "\nResults (second matrix): \n\n";
  th.printSummary();
  if (is_ir) {
    th.printIrSummary(&FGMRES);
  }
  error_sum += th.checkResult(1e-16);

  isTestPass(error_sum, test_name);

  // delete data on the heap
  delete A;
  delete [] rhs;

  return error_sum;
}

