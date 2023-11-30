#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;


  //we want error sum to be 0 at the end
  //that means PASS.
  //otheriwse it is a FAIL.
  int error_sum = 0;
  int status;
  const std::string data_path = (argc == 2) ? argv[1] : "./";


  std::string matrixFileName = data_path + "data/SiO2.mtx";
  std::string rhsFileName = data_path + "data/SiO2_rhs.mtx";


  ReSolve::matrix::Coo* A_coo;
  ReSolve::matrix::Csr* A;
  
  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA();
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;

  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);

  ReSolve::LinSolverDirectCuSparseILU0* Rf = new ReSolve::LinSolverDirectCuSparseILU0(workspace_CUDA);
  ReSolve::LinSolverIterativeRandFGMRES* FGMRES = new ReSolve::LinSolverIterativeRandFGMRES(matrix_handler, vector_handler,ReSolve::LinSolverIterativeRandFGMRES::cs, GS, "cuda");

  std::ifstream mat_file(matrixFileName);
  if(!mat_file.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName << "\n";
    return -1;
  }
  
  std::ifstream rhs_file(rhsFileName);
  if(!rhs_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName << "\n";
    return -1;
  }
  A_coo = ReSolve::io::readMatrixFromFile(mat_file);
  A = new ReSolve::matrix::Csr(A_coo->getNumRows(),
                               A_coo->getNumColumns(),
                               A_coo->getNnz(),
                               A_coo->symmetric(),
                               A_coo->expanded());

  rhs = ReSolve::io::readRhsFromFile(rhs_file);
 
  x = new real_type[A->getNumRows()];
  vec_rhs = new vector_type(A->getNumRows());
  vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(ReSolve::memory::HOST);
 
  //iinit guess is 0
  vec_x->allocate(ReSolve::memory::DEVICE);
  vec_x->setToZero(ReSolve::memory::DEVICE);

  mat_file.close();
  rhs_file.close();

  matrix_handler->coo2csr(A_coo,A, "cuda");
  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  
  real_type norm_b;
  matrix_handler->setValuesChanged(true, "cuda");

  status = Rf->setup(A);
  error_sum += status;
  
  FGMRES->setRestart(150);
  FGMRES->setMaxit(2500);
  FGMRES->setTol(1e-12);
  FGMRES->setup(A);
  
  status = GS->setup(FGMRES->getKrand(), FGMRES->getRestart()); 
  error_sum += status;

  //matrix_handler->setValuesChanged(true, "cuda");
  status = FGMRES->resetMatrix(A);
  error_sum += status;
  
  status = FGMRES->setupPreconditioner("LU", Rf);
  error_sum += status;
  
  FGMRES->setFlexible(1); 

  vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  FGMRES->solve(vec_rhs, vec_x);

  norm_b = vector_handler->dot(vec_rhs, vec_rhs, "cuda");
  norm_b = std::sqrt(norm_b);
  real_type final_norm_first =  FGMRES->getFinalResidualNorm();
  std::cout << "Randomized FGMRES results (first run): \n"
    << "\t Sketching method:                                    : CountSketch\n" 
    << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
    << std::scientific << std::setprecision(16) 
    << FGMRES->getInitResidualNorm()<<" \n"
    << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
    << FGMRES->getInitResidualNorm()/norm_b<<" \n"
    << "\t Final residual norm:            ||b-Ax||_2           : " 
    << FGMRES->getFinalResidualNorm() <<" \n"
    << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
    << FGMRES->getFinalResidualNorm()/norm_b <<" \n"
    << "\t Number of iterations                                 : " << FGMRES->getNumIter() << "\n";
  
  delete FGMRES;
  delete GS;
  GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);
  FGMRES = new ReSolve::LinSolverIterativeRandFGMRES(matrix_handler, vector_handler,ReSolve::LinSolverIterativeRandFGMRES::fwht, GS, "cuda");
  
  
  FGMRES->setRestart(150);
  FGMRES->setMaxit(2500);
  FGMRES->setTol(1e-12);
  FGMRES->setup(A);
  
  status = GS->setup(FGMRES->getKrand(), FGMRES->getRestart()); 
  error_sum += status;
  
  status = FGMRES->setupPreconditioner("LU", Rf);
  error_sum += status;
  
  vec_x->setToZero(ReSolve::memory::DEVICE);
  FGMRES->solve(vec_rhs, vec_x);


  std::cout << "Randomized FGMRES results (second run): \n"
    << "\t Sketching method:                                    : FWHT\n" 
    << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
    << std::scientific << std::setprecision(16) 
    << FGMRES->getInitResidualNorm()<<" \n"
    << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
    << FGMRES->getInitResidualNorm()/norm_b<<" \n"
    << "\t Final residual norm:            ||b-Ax||_2           : " 
    << FGMRES->getFinalResidualNorm() <<" \n"
    << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
    << FGMRES->getFinalResidualNorm()/norm_b <<" \n"
    << "\t Number of iterations                                 : " << FGMRES->getNumIter() << "\n";

  if ((error_sum == 0) && (final_norm_first/norm_b < 1e-11) && (FGMRES->getFinalResidualNorm()/norm_b < 1e-11 )) {
    std::cout<<"Test 5 (randomized GMRES) PASSED"<<std::endl<<std::endl;;
  } else {
    std::cout<<"Test 5 (randomized GMRES) FAILED, error sum: "<<error_sum<<std::endl<<std::endl;;
  }
  delete A;
  delete A_coo;
  delete Rf;
  delete [] x;
  delete [] rhs;
  delete vec_x;
  delete workspace_CUDA;
  delete matrix_handler;
  delete vector_handler;

  return error_sum;
}
