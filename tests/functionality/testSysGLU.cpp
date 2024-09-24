/**
 * @file testSysCuda.cpp
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
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>

#if defined (RESOLVE_USE_CUDA)
  using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
#elif defined (RESOLVE_USE_HIP)
  using workspace_type = ReSolve::LinAlgWorkspaceHIP;
#else
  using workspace_type = ReSolve::LinAlgWorkspaceCpu;
#endif

using namespace ReSolve::constants;
using namespace ReSolve::colors;

int main(int argc, char *argv[])
{
  // Use ReSolve data types.
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  int error_sum = 0; ///< error sum zero means test passes, otherwise fails
  int status = 0;

  workspace_type workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);

  ReSolve::SystemSolver solver(&workspace, "klu", "glu", "glu", "none", "none");

  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix
  std::ifstream mat1(matrixFileName1);
  if(!mat1.is_open()) {
    std::cout << "Failed to open file " << matrixFileName1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open()) {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  real_type* x   = new real_type[A->getNumRows()];
  vector_type* vec_rhs = new vector_type(A->getNumRows());
  vector_type* vec_x   = new vector_type(A->getNumRows());
  vector_type* vec_r   = new vector_type(A->getNumRows());
  rhs1_file.close();

  vec_x->allocate(ReSolve::memory::HOST);  //for KLU
  vec_x->allocate(ReSolve::memory::DEVICE);

  // Set RHS vector on CPU
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);

  // Set system matrix
  status = solver.setMatrix(A);
  error_sum += status;

  // Factorize the first matrix using KLU
  status = solver.analyze();
  error_sum += status;

  status = solver.factorize();
  error_sum += status;

  // but DO NOT SOLVE with KLU!
  status = solver.refactorizationSetup();
  error_sum += status;
  std::cout << "GLU setup status: " << status << std::endl;      

  // Move rhs vector data to GPU
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;
  std::cout << "GLU solve status: " << status << std::endl;      

  // Compute residual on device
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  
  // Compute residual norm
  real_type normRmatrix1 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  // Compute solution vector norm
  real_type normXtrue = sqrt(vector_handler.dot(vec_x, vec_x, ReSolve::memory::DEVICE));

  // Compute norm of the rhs vector
  real_type normB1 = sqrt(vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  // Compute norm of scaled residuals
  real_type inf_norm_A = 0.0;  
  matrix_handler.matrixInfNorm(A, &inf_norm_A, ReSolve::memory::DEVICE); 
  real_type inf_norm_x = vector_handler.infNorm(vec_x, ReSolve::memory::DEVICE);
  real_type inf_norm_r = vector_handler.infNorm(vec_r, ReSolve::memory::DEVICE);
  real_type nsr_norm   = inf_norm_r / (inf_norm_A * inf_norm_x);
  real_type nsr_system = solver.getNormOfScaledResiduals(vec_rhs, vec_x);
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

  // Create reference vectors for testing purposes
  vector_type* vec_test  = new vector_type(A->getNumRows());
  vector_type* vec_diff  = new vector_type(A->getNumRows());

  // Set the reference solution vector (all ones) on both, CPU and GPU
  real_type* x_data_ref = new real_type[A->getNumRows()];
  for (int i = 0; i < A->getNumRows(); ++i) {
    x_data_ref[i] = 1.0;
  }
  vec_test->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_test->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  //compute ||x_diff|| = ||x - x_true|| norm
  vec_diff->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vector_handler.axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  real_type normDiffMatrix1 = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));
 
  // Compute residual norm ON THE GPU using REFERENCE solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type exactSol_normRmatrix1 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  // Compute residual norm ON THE CPU using COMPUTED solution
  vec_x->update(vec_x->getData(ReSolve::memory::DEVICE), ReSolve::memory::DEVICE, ReSolve::memory::HOST);
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);
  error_sum += status;
  real_type normRmatrix1CPU = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::HOST));

  // Verify relative residual norm computation in SystemSolver
  real_type rel_residual_norm = solver.getResidualNorm(vec_rhs, vec_x);
  error = std::abs(normB1 * rel_residual_norm - normRmatrix1)/normRmatrix1;
  if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
    std::cout << "Relative residual norm computation failed:\n" << std::setprecision(16)
              << "\tTest value            : " << normRmatrix1/normB1 << "\n"
              << "\tSystemSolver computed : " << rel_residual_norm   << "\n\n";
    error_sum++;
  }
 
  std::cout << "Results (first matrix): \n\n" << std::scientific << std::setprecision(16);
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix1              << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2  (CPU)          : " << normRmatrix1CPU           << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix1/normB1       << " (relative residual norm)\n";
  std::cout << "\t ||b-A*x||/(||A||*||x||)     : " << nsr_norm                  << " (norm of scaled residuals)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix1           << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix1/normXtrue << " (relative solution error)\n";
  std::cout << "\t ||b-A*x_true||_2  (control) : " << exactSol_normRmatrix1     << " (residual norm with exact solution)\n\n";


  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open()) {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhsFileName2);
  if(!rhs2_file.is_open()) {
    std::cout << "Failed to open file " << rhsFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();

  // Update system values
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  // Refactorize and solve
  status = solver.refactorize();
  error_sum += status;
  std::cout << "CUSOLVER GLU refactorization status: " << status << std::endl;      

  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;

  // Compute residual norm for the second system
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type normRmatrix2 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));
  
  // Compute norm of the rhs vector for the second system
  real_type normB2 = sqrt(vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  // Compute norm of scaled residuals for the second system
  inf_norm_A = 0.0;  
  matrix_handler.matrixInfNorm(A, &inf_norm_A, ReSolve::memory::DEVICE); 
  inf_norm_x = vector_handler.infNorm(vec_x, ReSolve::memory::DEVICE);
  inf_norm_r = vector_handler.infNorm(vec_r, ReSolve::memory::DEVICE);
  nsr_norm   = inf_norm_r / (inf_norm_A * inf_norm_x);
  nsr_system = solver.getNormOfScaledResiduals(vec_rhs, vec_x);
  error      = std::abs(nsr_system - nsr_norm)/nsr_norm;

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

  //compute ||x_diff|| = ||x - x_true|| norm
  vec_diff->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vector_handler.axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  real_type normDiffMatrix2 = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));
 
  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type exactSol_normRmatrix2 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));
  
  // Verify relative residual norm computation in SystemSolver
  rel_residual_norm = solver.getResidualNorm(vec_rhs, vec_x);
  error = std::abs(normB2 * rel_residual_norm - normRmatrix2)/normRmatrix2;
  if (error > 10.0*std::numeric_limits<real_type>::epsilon()) {
    std::cout << "Relative residual norm computation failed:\n" << std::setprecision(16)
              << "\tTest value            : " << normRmatrix2/normB2 << "\n"
              << "\tSystemSolver computed : " << rel_residual_norm   << "\n\n";
    error_sum++;
  }
 
  std::cout << "Results (second matrix): " << std::endl << std::endl;
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix2              << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix2/normB2       << " (relative residual norm)\n";
  std::cout << "\t ||b-A*x||/(||A||*||x||)     : " << nsr_norm                  << " (norm of scaled residuals)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix2           << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix2/normXtrue << " (relative solution error)\n";
  std::cout << "\t ||b-A*x_true||_2  (control) : " << exactSol_normRmatrix2     << " (residual norm with exact solution)\n\n";

  if (!std::isfinite(normRmatrix1/normB1) || !std::isfinite(normRmatrix2/normB2)) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((normRmatrix1/normB1 > 1e-16 ) || (normRmatrix2/normB2 > 1e-16)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout << "Test KLU with cuSolverGLU refactorization " << GREEN << "PASSED" << CLEAR << std::endl;
  } else {
    std::cout << "Test KLU with cuSolverGLU refactorization " << RED << "FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl;
  }

  //now DELETE
  delete A;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete vec_diff;
  delete vec_rhs;
  delete vec_test;

  return error_sum;
}
