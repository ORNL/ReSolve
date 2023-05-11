#include "resolveMatrixIO.hpp"
#include "resolveMatrix.hpp"
#include <string>
#include <iostream>

int main( int argc, char *argv[] ){

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  int numSystems = atoi(argv[3]);
  printf("Family mtx file name: %s, total number of systems %d  \n", matrixFileName, numSystems);
  printf("Family rhs file name: %s, total number of systems %d  \n", rhsFileName, numSystems);

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::resolveMatrixIO* reader = new ReSolve::resolveMatrixIO;
  ReSolve::resolveMatrix* A;
  for (int i = 0; i < numSystems; ++i)
  {
    int j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];
    matrixFileNameFull = "";
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    std::cout<<"Reading: "<<matrixFileNameFull<<std::endl;
    if (i == 0) {
      std::cout<<"First matrix in the series!"<<std::endl;
      A = reader->readMatrixFromFile(matrixFileNameFull);
    }
    else {
    reader->readAndUpdateMatrix(matrixFileNameFull, A);
    }
    printf("Finished reading the matrix, sizes: %d %d %d\n", A->getNumRows(), A->getNumColumns(), A->getNnz());

 
  }

  return 0;
}
