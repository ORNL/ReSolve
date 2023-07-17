#include <string>
#include <iostream>

#include <resolve/MatrixCOO.hpp>
#include <resolve/MatrixCSR.hpp>
#include <resolve/MatrixCSC.hpp>
#include <resolve/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/MatrixHandler.hpp>
#include <resolve/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>


int main(Int argc, char *argv[] ){

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  Int numSystems = atoi(argv[3]);
  std::cout<<"Family mtx file name: "<< matrixFileName << ", total number of matrices: "<<numSystems<<std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << ", total number of RHSes: " << numSystems<<std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::MatrixCOO* A_coo;
  ReSolve::MatrixCSR* A;
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

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

  for (int i = 0; i < numSystems; ++i)
  {
    Int j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << "Reading: " << matrixFileNameFull << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << std::endl;
    // Read first matrix
    std::ifstream mat_file(matrixFileNameFull);
    if(!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrixFileNameFull << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhsFileNameFull);
    if(!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhsFileNameFull << "\n";
      return -1;
    }
    if (i == 0) {
      A_coo = ReSolve::matrix::io::readMatrixFromFile(mat_file);
      A = new ReSolve::MatrixCSR(A_coo->getNumRows(), A_coo->getNumColumns(), A_coo->getNnz(), A_coo->expanded(), A_coo->symmetric());

      rhs = ReSolve::matrix::io::readRhsFromFile(rhs_file);
      x = new Real[A->getNumRows()];
      vec_rhs = new ReSolve::Vector(A->getNumRows());
      vec_x = new ReSolve::Vector(A->getNumRows());
      vec_r = new ReSolve::Vector(A->getNumRows());
    }
    else {
      ReSolve::matrix::io::readAndUpdateMatrix(mat_file, A_coo);
      ReSolve::matrix::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    if (i < 2) { 
      matrix_handler->coo2csr(A_coo, A, "cpu");
      vec_rhs->update(rhs, "cpu", "cpu");
      vec_rhs->setDataUpdated("cpu");
    } else { 
      matrix_handler->coo2csr(A_coo, A, "cuda");
      vec_rhs->update(rhs, "cpu", "cuda");
    }
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnzExpanded()<<std::endl;
    //Now call direct solver
    if (i == 0) {
      KLU->setupParameters(1, 0.1, false);
    }
    int status;
    if (i < 2){
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
    } else {
  status =  KLU->refactorize();
      std::cout<<"KLU re-factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
    }
    vec_r->update(rhs, "cpu", "cuda");

    matrix_handler->setValuesChanged(true);

    matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "csr", "cuda"); 
    Real* test = vec_r->getData("cpu");

    printf("\t 2-Norm of the residual: %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda")));


  }

  return 0;
}
