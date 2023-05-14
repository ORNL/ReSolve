#include "resolveMatrixIO.hpp"
#include "resolveMatrix.hpp"
#include "resolveMatrixHandler.hpp"
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
  ReSolve::resolveMatrixHandler* matrix_handler =  new ReSolve::resolveMatrixHandler;
  resolveReal* rhs;
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
    }
    else {
      reader->readAndUpdateMatrix(matrixFileNameFull, A);
      reader->readAndUpdateRhs(rhsFileNameFull, rhs);
    }
    printf("Finished reading the matrix and rhs, sizes: %d %d %d\n", A->getNumRows(), A->getNumColumns(), A->getNnz());
    //Now convert to CSR.
    
    matrix_handler->coo2csr(A, "gpu");
    //Now call direct solver




  }

  return 0;
}
