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
      resolveInt i = 0;
      resolveInt m, n, nnz;
      bool symmetric = false;
      bool expanded = true;
      std::getline(file, line);
      //symmetric?
      size_t found = line.find("symmetric");
      if (found != std::string::npos) {
        symmetric = true;
        expanded = false;
      } 
      printf("symmetric? %d \n",symmetric );
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m >> nnz;
      std::cout<<"Matrix size: "<<n<<" x "<<m<<", nnz: "<<nnz<<std::endl; 
      //create matrix object
      resolveMatrix* A = new resolveMatrix(n, m, nnz,symmetric, expanded );  
      //create coo arrays
      resolveInt* coo_rows = new resolveInt[nnz];
      resolveInt* coo_cols = new resolveInt[nnz];
      resolveReal* coo_vals = new resolveReal[nnz];
      i = 0;
      resolveInt a, b;
      resolveReal c;
      while (file >> a>>b>>c){
        coo_rows[i] = a - 1;
        coo_cols[i] = b - 1;
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


  resolveReal* resolveMatrixIO::readRhsFromFile(std::string filename)
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {

      std::string line;
      resolveInt i = 0;
      resolveInt n, m;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m ;

      resolveReal* vec = new resolveReal[n];
      resolveReal a;
      while (file >> a){
        vec[i] = a;
        i++;
      }
      printf("VEC has %d elements \n", i);
      return vec;
      file.close();
    } else { 
      std::cout<<"Error opening file"<<std::endl;
      return nullptr;
    }

  }


  void resolveMatrixIO::readAndUpdateMatrix(std::string filename, resolveMatrix* A)
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {
	    A->setExpanded(false);
      std::string line;
      resolveInt i = 0;
      resolveInt m, n, nnz;
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
      resolveInt* coo_rows = A->getCooRowIndices("cpu");
      resolveInt* coo_cols = A->getCooColIndices("cpu");
      resolveReal* coo_vals = A->getCooValues("cpu");
      i = 0;
      resolveInt a, b;
      resolveReal c;
      while (file >> a>>b>>c){
        coo_rows[i] = a - 1;
        coo_cols[i] = b - 1;
        coo_vals[i] = c;
        i++;
      }
      file.close();
    } else {
      std::cout<<"Error opening file"<<std::endl;
    }

  }

  resolveReal* resolveMatrixIO::readAndUpdateRhs(std::string filename, resolveReal* rhs) 
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {

      std::string line;
      resolveInt i = 0;
      resolveInt n, m;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m ;

      if(rhs == nullptr) { 
        resolveReal* rhs = new resolveReal[n];
      } 
      resolveReal a;
      while (file >> a){
        rhs[i] = a;
        i++;
      }
      printf("VEC has %d elements \n", i);
      file.close();
    } else { 
      std::cout<<"Error opening file"<<std::endl;
    }

  }

  resolveInt resolveMatrixIO::writeMatrixToFile(resolveMatrix* A, std::string filename)
  {
  }

  resolveInt resolveMatrixIO::writeVectorToFile(resolveReal* x, std::string filename)
  {
  }
}


