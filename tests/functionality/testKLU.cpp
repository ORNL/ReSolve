#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/MatrixIO.hpp>
#include <resolve/Matrix.hpp>
#include <resolve/Vector.hpp>
#include <resolve/MatrixHandler.hpp>
#include <resolve/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
//author: KS
//functionality test to check whether KLU works correctly.

int main(Int argc, char *argv[] )
{
  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status = 0;

  ReSolve::Matrix* A;
  ReSolve::MatrixIO* reader = new ReSolve::MatrixIO;
  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);


  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  Real* rhs;
  Real* x;

  ReSolve::Vector* vec_rhs;
  ReSolve::Vector* vec_x;
  ReSolve::Vector* vec_r;

  Real one = 1.0;
  Real minusone = -1.0;
  Real zero = 0.0;
  
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  KLU->setupParameters(1, 0.1, false);

  // Input to this code is location of `data` directory where matrix files are stored
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName1 = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrixFileName2 = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhsFileName1 = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhsFileName2 = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  // Read first matrix

  A = reader->readMatrixFromFile(matrixFileName1);
  rhs = reader->readRhsFromFile(rhsFileName1);
  x = new Real[A->getNumRows()];
  vec_rhs = new ReSolve::Vector(A->getNumRows());
  vec_x = new ReSolve::Vector(A->getNumRows());
  vec_r = new ReSolve::Vector(A->getNumRows());

  // Convert first matrix to CSR format

  matrix_handler->coo2csr(A, "cpu");
  vec_rhs->update(rhs, "cpu", "cpu");
  vec_rhs->setDataUpdated("cpu");

  // Solve the first system using KLU
  status = KLU->setup(A);
  error_sum += status;

  status = KLU->analyze();
  error_sum += status;

  status = KLU->factorize();
  error_sum += status;

  status = KLU->solve(vec_rhs, vec_x);
  error_sum += status;


  ReSolve::Vector* vec_test;
  ReSolve::Vector* vec_diff;
  vec_test  = new ReSolve::Vector(A->getNumRows());
  vec_diff  = new ReSolve::Vector(A->getNumRows());
  Real* x_data = new Real[A->getNumRows()];
  for (int i=0; i<A->getNumRows(); ++i){
    x_data[i] = 1.0;
  }

  vec_test->setData(x_data, "cpu");
  vec_r->update(rhs, "cpu", "cuda");
  vec_diff->update(x_data, "cpu", "cuda");

  Real normXmatrix1 = sqrt(vector_handler->dot(vec_test, vec_test, "cuda"));
  matrix_handler->setValuesChanged(true);
  status = matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "cuda"); 
  error_sum += status;
  
  Real normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, "cuda"));


  //for testing only - control
  
  Real normXtrue = sqrt(vector_handler->dot(vec_x, vec_x, "cuda"));
  Real normB1 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, "cuda"));
  
  //compute x-x_true
  vector_handler->axpy(&minusone, vec_x, vec_diff, "cuda");
  //evaluate its norm
  Real normDiffMatrix1 = sqrt(vector_handler->dot(vec_diff, vec_diff, "cuda"));
 
  //compute the residual using exact solution
  vec_r->update(rhs, "cpu", "cuda");
  status = matrix_handler->matvec(A, vec_test, vec_r, &one, &minusone, "cuda"); 
  error_sum += status;
  Real exactSol_normRmatrix1 = sqrt(vector_handler->dot(vec_r, vec_r, "cuda"));
  
  std::cout<<"Results (first matrix): "<<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : "<<std::setprecision(16)<<normRmatrix1<<" (residual norm)"<<std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : "<<normRmatrix1/normB1<<" (scaled residual norm)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2              : "<<normDiffMatrix1<<" (solution error)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : "<<normDiffMatrix1/normXtrue<<" (scaled solution error)"<<std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : "<<exactSol_normRmatrix1<<" (control; residual norm with exact solution)"<<std::endl<<std::endl;
 // Load the second matrix

  reader->readAndUpdateMatrix(matrixFileName2, A);
  reader->readAndUpdateRhs(rhsFileName2, rhs);

  matrix_handler->coo2csr(A, "cuda");
  vec_rhs->update(rhs, "cpu", "cuda");

  // and solve it too
  status =  KLU->refactorize();
  error_sum += status;

  status = KLU->solve(vec_rhs, vec_x);
  error_sum += status;

  vec_r->update(rhs, "cpu", "cuda");
  matrix_handler->setValuesChanged(true);

  status = matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "cuda"); 
  error_sum += status;

  Real normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, "cuda"));
  
  //for testing only - control
  Real normB2 = sqrt(vector_handler->dot(vec_rhs, vec_rhs, "cuda"));
  //compute x-x_true
  vec_diff->update(x_data, "cpu", "cuda");
  vector_handler->axpy(&minusone, vec_x, vec_diff, "cuda");
  //evaluate its norm
  Real normDiffMatrix2 = sqrt(vector_handler->dot(vec_diff, vec_diff, "cuda"));
 
  //compute the residual using exact solution
  vec_r->update(rhs, "cpu", "cuda");
  status = matrix_handler->matvec(A, vec_test, vec_r, &one, &minusone, "cuda"); 
  error_sum += status;
  Real exactSol_normRmatrix2 = sqrt(vector_handler->dot(vec_r, vec_r, "cuda"));
  
  std::cout<<"Results (second matrix): "<<std::endl<<std::endl;
  std::cout<<"\t ||b-A*x||_2                 : "<<normRmatrix2<<" (residual norm)"<<std::endl;
  std::cout<<"\t ||b-A*x||_2/||b||_2         : "<<normRmatrix2/normB2<<" (scaled residual norm)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2              : "<<normDiffMatrix2<<" (solution error)"<<std::endl;
  std::cout<<"\t ||x-x_true||_2/||x_true||_2 : "<<normDiffMatrix2/normXtrue<<" (scaled solution error)"<<std::endl;
  std::cout<<"\t ||b-A*x_exact||_2           : "<<exactSol_normRmatrix2<<" (control; residual norm with exact solution)"<<std::endl<<std::endl;



  if ((error_sum == 0) && (normRmatrix1/normB1 < 1e-16 ) && (normRmatrix2/normB2 < 1e-16)) {
    std::cout<<"Test 1 (KLU with KLU refactorization) PASSED"<<std::endl;
  } else {

    std::cout<<"Test 1 (KLU with KLU refactorization) FAILED"<<std::endl;
  }
  return error_sum;
}
