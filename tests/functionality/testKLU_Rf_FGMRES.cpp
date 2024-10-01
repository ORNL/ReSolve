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
//author: KS
//functionality test to check whether cuSolverRf/FGMRES works correctly.

#if defined (RESOLVE_USE_CUDA)
#include <resolve/LinSolverDirectCuSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
  using solver_type    = ReSolve::LinSolverDirectCuSolverRf;
  std::string memory_space("cuda");
#elif defined (RESOLVE_USE_HIP)
#include <resolve/LinSolverDirectRocSolverRf.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceHIP;
  using solver_type    = ReSolve::LinSolverDirectRocSolverRf;
  std::string memory_space("hip");
#else
  using workspace_type = ReSolve::LinAlgWorkspaceCpu;
  using solver_type    = ReSolve::LinSolverDirectKLU;
  std::string memory_space("cpu");
#endif

using namespace ReSolve::constants;
using namespace ReSolve::colors;

int main(int argc, char *argv[])
{
  // Use ReSolve data types.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status = 0;

  workspace_type workspace;
  workspace.initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(&workspace);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(&workspace);

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

  solver_type* Rf = new solver_type(&workspace);
  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);
  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg2000_AC_00.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg2000_AC_02.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg2000_AC_00.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg2000_AC_02.mtx.ones";



  // Read first matrix
  std::ifstream mat1(matrixFileName1);
  if(!mat1.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName1 << "\n";
    return -1;
  }
  ReSolve::matrix::Csr* A = ReSolve::io::createCsrFromFile(mat1);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::createArrayFromFile(rhs1_file);
  real_type* x   = new real_type[A->getNumRows()];
  vector_type* vec_rhs = new vector_type(A->getNumRows());
  vector_type* vec_x   = new vector_type(A->getNumRows());
  vector_type* vec_r   = new vector_type(A->getNumRows());
  rhs1_file.close();

  // Convert first matrix to CSR format
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);

  // Solve the first system using KLU
  status = KLU->setup(A);
  error_sum += status;

  status = KLU->analyze();
  error_sum += status;

  status = KLU->factorize();
  error_sum += status;

  status = KLU->solve(vec_rhs, vec_x);
  error_sum += status;

  vector_type* vec_test;
  vector_type* vec_diff;

  vec_test  = new vector_type(A->getNumRows());
  vec_diff  = new vector_type(A->getNumRows());
  real_type* x_data = new real_type[A->getNumRows()];

  for (int i=0; i<A->getNumRows(); ++i){
    x_data[i] = 1.0;
  }

  vec_test->setData(x_data, ReSolve::memory::HOST);
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

  matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
  //evaluate the residual ||b-Ax||
  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;

  real_type normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));


  //for testing only - control
  real_type normXtrue = sqrt(vector_handler->dot(vec_x, vec_x, ReSolve::memory::DEVICE));
  real_type normB1 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));

  //compute x-x_true
  vector_handler->axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  //evaluate its norm
  real_type normDiffMatrix1 = sqrt(vector_handler->dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));

  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = matrix_handler->matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type exactSol_normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  //evaluate the residual ON THE CPU using COMPUTED solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);
  error_sum += status;

  real_type normRmatrix1CPU = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));

  std::cout<<"Results (first matrix): " << std::scientific <<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : " << std::setprecision(16) << normRmatrix1    << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2  (CPU)          : " << std::setprecision(16) << normRmatrix1CPU << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : " << normRmatrix1/normB1   << " (scaled residual norm)"             << std::endl;
  std::cout<<"\t ||x-x_true||_2              : " << normDiffMatrix1       << " (solution error)"                   << std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix1/normXtrue << " (scaled solution error)"        << std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix1 << " (control; residual norm with exact solution)\n\n";


  // Now prepare the Rf solver

  ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*) KLU->getLFactor();
  ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*) KLU->getUFactor();
  ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
  ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
  error_sum += matrix_handler->csc2csr(L_csc,L, ReSolve::memory::DEVICE);
  error_sum += matrix_handler->csc2csr(U_csc,U, ReSolve::memory::DEVICE);

  if (L == nullptr) {
    printf("ERROR");
  }
  index_type* P = KLU->getPOrdering();
  index_type* Q = KLU->getQOrdering();
  error_sum += Rf->setup(A, L, U, P, Q); 

  FGMRES->setMaxit(400); 
  FGMRES->setRestart(100); 
  FGMRES->setFlexible(1); 
  FGMRES->setTol(1e-17); 

  status =  FGMRES->setup(A); 
  error_sum += status;

  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::updateMatrixFromFile(mat2, A);
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

  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  Rf->setNumericalProperties(1e-14, 1e-1);

  status = Rf->refactorize();
  error_sum += status;
  
  vec_x->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = Rf->solve(vec_x);
  error_sum += status;
  
  FGMRES->resetMatrix(A);
  status = FGMRES->setupPreconditioner("LU", Rf);
  error_sum += status;

  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = FGMRES->solve(vec_rhs, vec_x);
  error_sum += status;

  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

  //evaluate final residual
  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;

  real_type normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));


  //for testing only - control
  real_type normB2 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));
  //compute x-x_true
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  vector_handler->axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::DEVICE);
  //evaluate its norm
  real_type normDiffMatrix2 = sqrt(vector_handler->dot(vec_diff, vec_diff, ReSolve::memory::DEVICE));

  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  status = matrix_handler->matvec(A, vec_test, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
  error_sum += status;
  real_type exactSol_normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
  std::cout<<"Results (second matrix): "<<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : "<<normRmatrix2<<" (residual norm)"<<std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : "<<normRmatrix2/normB2<<" (scaled residual norm)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2              : "<<normDiffMatrix2<<" (solution error)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : "<<normDiffMatrix2/normXtrue<<" (scaled solution error)"<<std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : "<<exactSol_normRmatrix2<<" (control; residual norm with exact solution)"<<std::endl;
  std::cout<<"\t IR iterations               : "<<FGMRES->getNumIter()<<" (max 400, restart 100)"<<std::endl;
  std::cout<<"\t IR starting res. norm       : "<<FGMRES->getInitResidualNorm()<<" "<<std::endl;
  std::cout<<"\t IR final res. norm          : "<<FGMRES->getFinalResidualNorm()<<" "<<std::endl<<std::endl;

  if (!std::isfinite(normRmatrix1/normB1) || !std::isfinite(normRmatrix2/normB2)) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((normRmatrix1/normB1 > 1e-12 ) || (normRmatrix2/normB2 > 1e-15)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout << "Test KLU with cuSolverGLU refactorization " << GREEN << "PASSED" << CLEAR << std::endl;
  } else {
    std::cout << "Test KLU with cuSolverGLU refactorization " << RED << "FAILED" << CLEAR
              << ", error sum: " << error_sum << std::endl;
  }

  delete A;
  delete KLU;
  delete GS;
  delete FGMRES;
  delete Rf;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return error_sum;
}
