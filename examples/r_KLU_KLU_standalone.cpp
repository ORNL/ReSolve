#include <string>
#include <iostream>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>


int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  std::cout<<"Family mtx file name: "<< matrixFileName << std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << std::endl;

  std::string fileId;
  std::string rhsId;

  ReSolve::matrix::Coo* A_coo;
  ReSolve::matrix::Csr* A;
  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  real_type* rhs;
  real_type* x;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;

  real_type one = 1.0;
  real_type minusone = -1.0;

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;



    // Read matrix first
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << "Reading: " << matrixFileName << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << std::endl;
    // Read first matrix
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
      vec_r = new vector_type(A->getNumRows());
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    matrix_handler->coo2csr(A_coo, A, "cpu");
    vec_rhs->update(rhs, "cpu", "cpu");
    vec_rhs->setDataUpdated("cpu");
    std::cout << "COO to CSR completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;
    //Now call direct solver
    KLU->setupParameters(1, 0.1, false);
    int status;
    KLU->setup(A);
    status = KLU->analyze();
    std::cout<<"KLU analysis status: "<<status<<std::endl;
    status = KLU->factorize();
    std::cout << "KLU factorization status: " << status << std::endl;
    status = KLU->solve(vec_rhs, vec_x);
    std::cout << "KLU solve status: " << status << std::endl;      
    vec_r->update(rhs, "cpu", "cuda");

    matrix_handler->setValuesChanged(true);

    matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "csr", "cuda"); 
    real_type* test = vec_r->getData("cpu");

    printf("\t 2-Norm of the residual: %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda")));



  //now DELETE
  delete A;
  delete KLU;
  delete x;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
