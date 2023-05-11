#include <fstream>
#include <iostream>
#include <sstream>
#include "resolveMatrixIO.hpp"

namespace ReSolve
{

  resolveMatrixIO::resolveMatrixIO(){};

  resolveMatrixIO::~resolveMatrixIO()
  {
  }

  resolveMatrix* resolveMatrixIO::readMatrixFromFile(std::string filename)
  {
    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {
      std::string line;
      int i = 0;
      long  int m, n, nnz;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
       // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m >> nnz;
      std::cout<<"Matrix size: "<<n<<" x "<<m<<", nnz: "<<nnz<<std::endl; 
      //create matrix object
      resolveMatrix* A = new resolveMatrix(n, m, nnz);  
      //create coo arrays
      int* coo_rows = new int[nnz];
      int* coo_cols = new int[nnz];
      double* coo_vals = new double[nnz];
      i = 0;
      int a, b;
      double c;
      while (file >> a>>b>>c){
        coo_rows[i] = a;
        coo_cols[i] = b;
        coo_vals[i] = c;

        i++;
      }
      A->setCoo(coo_rows, coo_cols, coo_vals, "cpu");
      return A;
      file.close();
    } else {
      std::cout<<"Error opening file"<<std::endl;
      return nullptr;
    }
  }


  double* resolveMatrixIO::readRhsFromFile(std::string filename)
  {

  } 

  void resolveMatrixIO::readAndUpdateMatrix(std::string filename, resolveMatrix* A)
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {
      std::string line;
      int i = 0;
      long  int m, n, nnz;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
      //  std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m >> nnz;
      std::cout<<"Matrix size: "<<n<<" x "<<m<<", nnz: "<<nnz<<std::endl; 
      if ((A->getNumRows() != n) || (A->getNumColumns() != m) || (A->getNnz() != nnz)){
        printf("Wrong matrix size! \n ");
        exit(0);
      }
      //create coo arrays
      int* coo_rows = A->getCooRowIndices("cpu");
      int* coo_cols = A->getCooColIndices("cpu");
      double* coo_vals = A->getCooValues("cpu");
      i = 0;
      int a, b;
      double c;
      while (file >> a>>b>>c){
        coo_rows[i] = a;
        coo_cols[i] = b;
        coo_vals[i] = c;
        i++;
      }
      file.close();
    } else {
      std::cout<<"Error opening file"<<std::endl;
    }

  }

  double* resolveMatrixIO::readAndUpdateRhs(std::string filename, double* rhs) 
  {
  }

  int resolveMatrixIO::writeMatrixToFile(resolveMatrix* A, std::string filename)
  {
  }

  int resolveMatrixIO::writeVectorToFile(double* x, std::string filename)
  {
  }
}


