#include "resolveMatrixIO.hpp"
#include "resolveMatrix.hpp"
#include "resolveVector.hpp"
#include "resolveMatrixHandler.hpp"
#include "resolveVectorHandler.hpp"
#include "resolveLinSolverDirectKLU.hpp"
#include <string>
#include <iostream>

int main(resolveInt argc, char *argv[] ){

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  resolveInt numSystems = atoi(argv[3]);
  std::cout<<"Family mtx file name: "<< matrixFileName << ", total number of matrices: "<<numSystems<<std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << ", total number of RHSes: " << numSystems<<std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::resolveMatrixIO* reader = new ReSolve::resolveMatrixIO;
  ReSolve::resolveMatrix* A;
  ReSolve::resolveLinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::resolveLinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::resolveMatrixHandler* matrix_handler =  new ReSolve::resolveMatrixHandler(workspace_CUDA);
  ReSolve::resolveVectorHandler* vector_handler =  new ReSolve::resolveVectorHandler(workspace_CUDA);
  resolveReal* rhs;
  resolveReal* x;

  ReSolve::resolveVector* vec_rhs;
  ReSolve::resolveVector* vec_x;
  ReSolve::resolveVector* vec_r;

  resolveReal one = 1.0;
  resolveReal minusone = -1.0;

  ReSolve::resolveLinSolverDirectKLU* KLU = new ReSolve::resolveLinSolverDirectKLU;

  for (int i = 0; i < numSystems; ++i)
  {
    resolveInt j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout<<std::endl<<std::endl<<std::endl;
    std::cout<<"========================================================================================================================"<<std::endl;
    std::cout<<"Reading: "<<matrixFileNameFull<<std::endl;
    std::cout<<"========================================================================================================================"<<std::endl;
    std::cout<<std::endl;
    if (i == 0) {
      A = reader->readMatrixFromFile(matrixFileNameFull);

      rhs = reader->readRhsFromFile(rhsFileNameFull);
      x = new resolveReal[A->getNumRows()];
      vec_rhs = new ReSolve::resolveVector(A->getNumRows());
      vec_x = new ReSolve::resolveVector(A->getNumRows());
      vec_r = new ReSolve::resolveVector(A->getNumRows());
    }
    else {
      reader->readAndUpdateMatrix(matrixFileNameFull, A);
      reader->readAndUpdateRhs(rhsFileNameFull, rhs);
    }
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;

    //Now convert to CSR.
    if (i < 2) { 
      matrix_handler->coo2csr(A, "cpu");
      vec_rhs->update(rhs, "cpu", "cpu");
      vec_rhs->setDataUpdated("cpu");
    } else { 
      matrix_handler->coo2csr(A, "cuda");
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
    }
    vec_r->update(rhs, "cpu", "cuda");


    matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "cuda"); 
    resolveReal* test = vec_r->getData("cpu");

    printf("\t 2-Norm of the residual: %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda")));


  }

  return 0;
}
