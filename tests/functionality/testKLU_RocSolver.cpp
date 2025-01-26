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
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/utilities/params/CliOptions.hpp>

#include "TestHelper.hpp"


static int runTest(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  return runTest(argc, argv);
}

int runTest(int argc, char *argv[])
{
  // Use ReSolve data types.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using matrix_type = ReSolve::matrix::Sparse;

  // Error sum needs to be zero at the end for test to pass.
  int error_sum = 0;
  int status = 0;

  // Collect all CLI
  // ReSolve::CliOptions options(argc, argv);
  // ReSolve::CliOptions::Option* opt = nullptr;

  // opt = options.getParamFromKey("-d");
  // std::string data_path = opt ? (*opt).second : "./";

  // opt = options.getParamFromKey("-g");
  // std::string gs = opt ? (*opt).second : "CGS2";

  // opt = options.getParamFromKey("-s");
  // std::string sketch = opt ? (*opt).second : "count";

  // opt = options.getParamFromKey("-x");
  // std::string flexible = opt ? (*opt).second : "yes";

  // Create workspace
  ReSolve::LinAlgWorkspaceHIP workspace;
  workspace.initializeHandles();

  // Create direct solvers
  ReSolve::LinSolverDirectKLU KLU;
  ReSolve::LinSolverDirectRocSolverRf Rf(&workspace);
  Rf.setSolveMode(0);

  // Create iterative solver
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);
  ReSolve::GramSchmidt GS(&vector_handler, ReSolve::GramSchmidt::CGS2);
  ReSolve::LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);
  FGMRES.setMaxit(200); 
  FGMRES.setRestart(100); 

  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix
  std::ifstream mat1(matrixFileName1);
  if(!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  A->syncData(ReSolve::memory::DEVICE);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
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

  status = Rf.solve(&vec_rhs, &vec_x);
  error_sum += status;

  status =  FGMRES.setup(A); 
  error_sum += status;

  status = FGMRES.setupPreconditioner("LU", &Rf);
  error_sum += status;

  status = FGMRES.solve(&vec_rhs, &vec_x);
  error_sum += status;

  // Setup test helper
  TestHelper<ReSolve::LinAlgWorkspaceHIP> th(A, &vec_rhs, &vec_x, workspace);

  // Print result summary and check solution
  std::cout << "Results (first matrix): \n\n";
  th.printSummary();
  error_sum += th.checkResult(1e-16);

  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  A->syncData(ReSolve::memory::DEVICE);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhsFileName2);
  if(!rhs2_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();
  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  status = Rf.refactorize();
  error_sum += status;
  std::cout << "rocSolverRf refactorization status: " << status << std::endl;      
  
  // status = Rf.solve(&vec_rhs, &vec_x);
  // error_sum += status;

  FGMRES.resetMatrix(A);
  status = FGMRES.setupPreconditioner("LU", &Rf);
  error_sum += status;

  status = FGMRES.solve(&vec_rhs, &vec_x);
  error_sum += status;

  th.resetSystem(A, &vec_rhs, &vec_x);

  std::cout << "Results (second matrix): \n\n";
  th.printSummary();
  error_sum += th.checkResult(1e-16);

  isTestPass(error_sum);

  // delete data on the heap
  delete A;
  delete [] rhs;

  return error_sum;
}

