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

using namespace ReSolve::constants;
using namespace ReSolve::colors;

int main(int argc, char *argv[])
{
  // Use ReSolve data types.
  using real_type   = ReSolve::real_type;
  using index_type  = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;

  // Error sum needs to be 0 at the end for test to PASS.
  // It is a FAIL otheriwse.
  int error_sum = 0;
  int status = 0;

  // Create workspace and initialize its handles.
  workspace_type workspace;
  workspace.initializeHandles();

  // Create linear algebra handlers
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);

  // Create system solver
  ReSolve::SystemSolver solver(&workspace);

  // Configure solver (CUDA-based solver needs slightly different
  // settings than HIP-based one)
  solver.setRefinementMethod("fgmres", "cgs2");
  solver.getIterativeSolver().setRestart(100);
  if (memory_space == "hip") {
    solver.getIterativeSolver().setMaxit(200);
  }
  if (memory_space == "cuda") {
    solver.getIterativeSolver().setMaxit(400);
    solver.getIterativeSolver().setTol(1e-17);
  }
  
  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg2000_AC_00.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg2000_AC_02.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg2000_AC_00.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg2000_AC_02.mtx.ones";


  // Read first matrix
  std::ifstream mat1(matrixFileName1);
  if(!mat1.is_open()) {
    std::cout << "Failed to open file " << matrixFileName1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1, true);
  A->syncData(ReSolve::memory::DEVICE);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open()) {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  rhs1_file.close();

  // Create rhs, solution and residual vectors
  vector_type* vec_rhs = new vector_type(A->getNumRows());
  vector_type* vec_x   = new vector_type(A->getNumRows());
  vector_type* vec_r   = new vector_type(A->getNumRows());

  // Allocate solution vector
  vec_x->allocate(ReSolve::memory::HOST);  //for KLU
  vec_x->allocate(ReSolve::memory::DEVICE);

  // Set RHS vector on CPU (update function allocates)
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);

  // Add system matrix to the solver
  status = solver.setMatrix(A);
  error_sum += status;

  // Solve the first system using KLU
  status = solver.analyze();
  error_sum += status;

  status = solver.factorize();
  error_sum += status;

  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;

  // Evaluate the residual norm ||b-Ax|| on the device
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type normRmatrix1 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  // Compute norm of the rhs vector
  real_type normB1 = sqrt(vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  // Compute solution vector norm
  real_type normXtrue = sqrt(vector_handler.dot(vec_x, vec_x, ReSolve::memory::DEVICE));

  // Compute norm of scaled residuals
  real_type inf_norm_A = 0.0;  
  matrix_handler.matrixInfNorm(A, &inf_norm_A, ReSolve::memory::DEVICE); 
  real_type inf_norm_x = vector_handler.infNorm(vec_x, ReSolve::memory::DEVICE);
  real_type inf_norm_r = vector_handler.infNorm(vec_r, ReSolve::memory::DEVICE);
  real_type nsr_norm   = inf_norm_r / (inf_norm_A * inf_norm_x);
  real_type nsr_system = solver.getNormOfScaledResiduals(vec_rhs, vec_x);
  real_type error      = std::abs(nsr_system - nsr_norm)/nsr_norm;

  // Test norm of scaled residuals method in SystemSolver
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
  vector_type* vec_test = new vector_type(A->getNumRows());
  vector_type* vec_diff = new vector_type(A->getNumRows());

  // Set the reference solution vector (all ones) on both, CPU and GPU
  real_type* x_data_ref = new real_type[A->getNumRows()];
  for (int i=0; i<A->getNumRows(); ++i){
    x_data_ref[i] = 1.0;
  }
  vec_test->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vec_diff->update(x_data_ref, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  // compute ||x-x_true|| norm
  vector_handler.axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  real_type normDiffMatrix1 = sqrt(vector_handler.dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));

  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type exactSol_normRmatrix1 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  //evaluate the residual ON THE CPU using COMPUTED solution
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
 
  // Print out the result summary
  std::cout << std::scientific << std::setprecision(16);
  std::cout << "Results (first matrix): \n\n";
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix1              << " (residual norm)" << std::endl;
  std::cout << "\t ||b-A*x||_2  (CPU)          : " << normRmatrix1CPU           << " (residual norm)" << std::endl;
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix1/normB1       << " (relative residual norm)"  << std::endl;
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix1           << " (solution error)"          << std::endl;
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix1/normXtrue << " (relative solution error)" << std::endl;
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix1     << " (control; residual norm with exact solution)\n\n";


  // Now prepare the Rf solver
  status = solver.refactorizationSetup();
  error_sum += status;

  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open()) {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
  A->syncData(ReSolve::memory::DEVICE);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhsFileName2);
  if(!rhs2_file.is_open()) {
    std::cout << "Failed to open file " << rhsFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateArrayFromFile(rhs2_file, &rhs);
  rhs2_file.close();

  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  status = solver.refactorize();
  error_sum += status;
  
  vec_x->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = solver.solve(vec_rhs, vec_x);
  error_sum += status;
  
  // Compute residual norm for the second system
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);
  status = matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type normRmatrix2 = sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  //for testing only - control
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
 
  // Get solver parameters
  real_type tol = solver.getIterativeSolver().getTol();
  index_type restart = solver.getIterativeSolver().getRestart();
  index_type maxit = solver.getIterativeSolver().getMaxit();

  // Get solver stats
  index_type num_iter   = solver.getIterativeSolver().getNumIter();
  real_type init_rnorm  = solver.getIterativeSolver().getInitResidualNorm();
  real_type final_rnorm = solver.getIterativeSolver().getFinalResidualNorm();
  

  std::cout << "Results (second matrix): " << std::endl << std::endl;
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix2              << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix2/normB2       << " (relative residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix2           << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix2/normXtrue << " (relative solution error)\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix2     << " (control; residual norm with exact solution)\n";
  std::cout << "\t IR iterations               : " << num_iter    << " (max " << maxit << ", restart " << restart << ")\n";
  std::cout << "\t IR starting res. norm       : " << init_rnorm  << "\n";
  std::cout << "\t IR final res. norm          : " << final_rnorm << " (tol " << std::setprecision(2) << tol << ")\n\n";

  if (!std::isfinite(normRmatrix1/normB1) || !std::isfinite(normRmatrix2/normB2)) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((normRmatrix1/normB1 > 1e-12 ) || (normRmatrix2/normB2 > 1e-15)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout<<"Test KLU with Rf solver + IR " << GREEN << "PASSED" << CLEAR <<std::endl<<std::endl;;
  } else {
    std::cout<<"Test KLU with Rf solver + IR " << RED << "FAILED" << CLEAR << ", error sum: "<<error_sum<<std::endl<<std::endl;;
  }

  delete A;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete vec_rhs;
  delete vec_test;
  delete vec_diff;

  return error_sum;
}
