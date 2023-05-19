#include "resolveMatrixIO.hpp"
#include "resolveMatrix.hpp"
#include "resolveMatrixHandler.hpp"
#include "resolveLinSolverDirectKLU.hpp"
#include <string>
#include <iostream>

int main(resolveInt argc, char *argv[] ){

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  resolveInt numSystems = atoi(argv[3]);
  printf("Family mtx file name: %s, total number of systems %d  \n", matrixFileName, numSystems);
  printf("Family rhs file name: %s, total number of systems %d  \n", rhsFileName, numSystems);

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::resolveMatrixIO* reader = new ReSolve::resolveMatrixIO;
  ReSolve::resolveMatrix* A;
  ReSolve::resolveLinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::resolveLinAlgWorkspaceCUDA;
 workspace_CUDA->initializeHandles();
  ReSolve::resolveMatrixHandler* matrix_handler =  new ReSolve::resolveMatrixHandler(workspace_CUDA);
  resolveReal* rhs;
  resolveReal* x;
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
    std::cout<<"Reading: "<<matrixFileNameFull<<std::endl;
    if (i == 0) {
      std::cout<<"First matrix in the series!"<<std::endl;
      A = reader->readMatrixFromFile(matrixFileNameFull);

      std::cout<<"First rhs in the series!"<<std::endl;
      rhs = reader->readRhsFromFile(rhsFileNameFull);
      std::cout<<"Rhs allocated!"<<std::endl;
      x = new resolveReal[A->getNumRows()];
    }
    else {
      reader->readAndUpdateMatrix(matrixFileNameFull, A);
      reader->readAndUpdateRhs(rhsFileNameFull, rhs);
    }
    printf("Finished reading the matrix and rhs, sizes: %d %d %d\n", A->getNumRows(), A->getNumColumns(), A->getNnz());
  
    //Now convert to CSR.
   if (i < 2) { 
    matrix_handler->coo2csr(A, "cpu");
   } else { 
    matrix_handler->coo2csr(A, "gpu");
   }
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
     status = KLU->solve(rhs, x);
     std::cout<<"KLU solve status: "<<status<<std::endl;      
   }

  // matrix_handler->matvec(A, d_x, res 



  }

  return 0;
}
