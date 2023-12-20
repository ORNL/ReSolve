#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
//author: KS
//functionality test to check whether KLU works correctly.
using namespace ReSolve::constants;

int main(int argc, char *argv[])
{
  // Use ReSolve data types.
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status = 0;

  ReSolve::LinAlgWorkspaceCpu* workspace = new ReSolve::LinAlgWorkspaceCpu();
  ReSolve::MatrixHandler* matrix_handler = new ReSolve::MatrixHandler(workspace);
  ReSolve::VectorHandler* vector_handler = new ReSolve::VectorHandler(workspace);
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

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
  ReSolve::matrix::Coo* A_coo = ReSolve::io::readMatrixFromFile(mat1);
  ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(A_coo, ReSolve::memory::HOST);
  mat1.close();

  // Read first rhs vector
  std::ifstream rhs1_file(rhsFileName1);
  if(!rhs1_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName1 << "\n";
    return -1;
  }
  real_type* rhs = ReSolve::io::readRhsFromFile(rhs1_file);
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
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);

  // real_type normXmatrix1 = sqrt(vector_handler->dot(vec_test, vec_test, ReSolve::memory::HOST));
  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);
  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE,"csr",ReSolve::memory::HOST); 
  error_sum += status;
  
  real_type normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST));


  //for testing only - control
  
  real_type normXtrue = sqrt(vector_handler->dot(vec_x, vec_x, ReSolve::memory::HOST));
  real_type normB1 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::HOST));
  
  //compute x-x_true
  vector_handler->axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::HOST);
  //evaluate its norm
  real_type normDiffMatrix1 = sqrt(vector_handler->dot(vec_diff, vec_diff, ReSolve::memory::HOST));
 
  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  status = matrix_handler->matvec(A, vec_test, vec_r, &ONE, &MINUSONE,"csr", ReSolve::memory::HOST); 
  error_sum += status;
  real_type exactSol_normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST));
  //evaluate the residual ON THE CPU using COMPUTED solution
 
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE,"csr", ReSolve::memory::HOST);
  error_sum += status;
 
  real_type normRmatrix1CPU = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST));
 
  std::cout<<"Results (first matrix): "<<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : " << std::setprecision(16) << normRmatrix1    << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2  (CPU)          : " << std::setprecision(16) << normRmatrix1CPU << " (residual norm)" << std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : " << normRmatrix1/normB1   << " (scaled residual norm)"             << std::endl;
  std::cout<<"\t ||x-x_true||_2              : " << normDiffMatrix1       << " (solution error)"                   << std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix1/normXtrue << " (scaled solution error)"        << std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix1 << " (control; residual norm with exact solution)\n\n";

  // Load the second matrix
  std::ifstream mat2(matrixFileName2);
  if(!mat2.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName2 << "\n";
    return -1;
  }
  ReSolve::io::readAndUpdateMatrix(mat2, A_coo);
  mat2.close();

  // Load the second rhs vector
  std::ifstream rhs2_file(rhsFileName2);
  if(!rhs2_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName2 << "\n";
    return -1;
  }
  ReSolve::io::readAndUpdateRhs(rhs2_file, &rhs);
  rhs2_file.close();

  A->updateFromCoo(A_coo, ReSolve::memory::HOST);
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  // and solve it too
  status =  KLU->refactorize();
  error_sum += status;

  status = KLU->solve(vec_rhs, vec_x);
  error_sum += status;

  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

  status = matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, "csr", ReSolve::memory::HOST); 
  error_sum += status;

  real_type normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST));
  
  //for testing only - control
  real_type normB2 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::HOST));
  //compute x-x_true
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vector_handler->axpy(&MINUSONE, vec_x, vec_diff, ReSolve::memory::HOST);
  //evaluate its norm
  real_type normDiffMatrix2 = sqrt(vector_handler->dot(vec_diff, vec_diff, ReSolve::memory::HOST));
 
  //compute the residual using exact solution
  vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  status = matrix_handler->matvec(A, vec_test, vec_r, &ONE, &MINUSONE, "csr", ReSolve::memory::HOST); 
  error_sum += status;
  real_type exactSol_normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST));
  
  std::cout<<"Results (second matrix): "<<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : "<<normRmatrix2<<" (residual norm)"<<std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : "<<normRmatrix2/normB2<<" (scaled residual norm)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2              : "<<normDiffMatrix2<<" (solution error)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : "<<normDiffMatrix2/normXtrue<<" (scaled solution error)"<<std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : "<<exactSol_normRmatrix2<<" (control; residual norm with exact solution)"<<std::endl<<std::endl;

  if ((normRmatrix1/normB1 > 1e-16 ) || (normRmatrix2/normB2 > 1e-16)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout<<"Test KLU with KLU refactorization PASSED"<<std::endl;
  } else {
    std::cout<<"Test KLU with KLU refactorization FAILED, error sum: "<<error_sum<<std::endl;
  }

  //now DELETE
  delete A;
  delete KLU;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return error_sum;
}
