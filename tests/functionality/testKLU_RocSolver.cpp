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
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include "TestHelper.hpp"

using namespace ReSolve::constants;
using namespace ReSolve::colors;


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

  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status = 0;

  std::cout << "REFACTORING IN PROGRESS!\n";

  ReSolve::LinAlgWorkspaceHIP workspace_HIP;
  workspace_HIP.initializeHandles();

  ReSolve::LinSolverDirectKLU KLU;
  ReSolve::LinSolverDirectRocSolverRf Rf(&workspace_HIP);

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

  status = KLU.solve(&vec_rhs, &vec_x);
  error_sum += status;

  std::cout << "KLU solve status: " << status <<std::endl;      
  TestHelper<ReSolve::LinAlgWorkspaceHIP> th(A, &vec_rhs, &vec_x, workspace_HIP);

  matrix_type* L = KLU.getLFactor();
  matrix_type* U = KLU.getUFactor();
  if (L == nullptr) {
    std::cout << "ERROR";
  }
  index_type* P = KLU.getPOrdering();
  index_type* Q = KLU.getQOrdering();

  status = Rf.setup(A, L, U, P, Q, &vec_rhs); 
  error_sum += status;
  std::cout << "Rf setup status: " << status << std::endl;      

  status = Rf.refactorize();
  error_sum += status;

  // Print result summary and check solution
  std::cout << "Results (first matrix): \n\n";
  th.printSummary();
  if (!std::isfinite(th.getNormResidualScaled())) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((th.getNormResidualScaled() > 1e-16 )) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }

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
  
  status = Rf.solve(&vec_rhs, &vec_x);
  error_sum += status;

  th.resetSystem(A, &vec_rhs, &vec_x);

  std::cout<<"Results (second matrix): "<<std::endl<<std::endl;
  th.printSummary();

  if (!std::isfinite(th.getNormResidualScaled())) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((th.getNormResidualScaled() > 1e-16 )) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }

  if (error_sum == 0) {
    std::cout << "Test KLU with rocsolverRf refactorization " << GREEN << "PASSED" << CLEAR << std::endl;
  } else {
    std::cout << "Test KLU with rocsolverRf refactorization " << RED << "FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl;
  }

  // delete data on the heap
  delete A;
  delete [] rhs;

  return error_sum;
}

