#include <fstream>
#include <iostream>
#include <sstream>
#include "MatrixIO.hpp"

namespace ReSolve
{

  MatrixIO::MatrixIO(){};

  MatrixIO::~MatrixIO()
  {
  }

  Matrix* MatrixIO::readMatrixFromFile(std::string filename)
  {
    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {
      std::string line;
      Int i = 0;
      Int m, n, nnz;
      bool symmetric = false;
      bool expanded = true;
      std::getline(file, line);
      //symmetric?
      size_t found = line.find("symmetric");
      if (found != std::string::npos) {
        symmetric = true;
        expanded = false;
      } 
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m >> nnz;
      //create matrix object
      Matrix* A = new Matrix(n, m, nnz,symmetric, expanded );  
      //create coo arrays
      Int* coo_rows = new Int[nnz];
      Int* coo_cols = new Int[nnz];
      Real* coo_vals = new Real[nnz];
      i = 0;
      Int a, b;
      Real c;
      while (file >> a>>b>>c){
        coo_rows[i] = a - 1;
        coo_cols[i] = b - 1;
        coo_vals[i] = c;

        i++;
      }
      A->setCoo(coo_rows, coo_cols, coo_vals, "cpu");
      return A;
      file.close();
    }
    std::cout<<"Error opening file"<<std::endl;
    return nullptr;
  }



  Real* MatrixIO::readRhsFromFile(std::string filename)
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {

      std::string line;
      Int i = 0;
      Int n, m;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m ;
      Real* vec = new Real[n];
      Real a;
      while (file >> a){
        vec[i] = a;
        i++;
      }
      file.close();
      return vec;
    }
    std::cout<<"Error opening file"<<std::endl;
    return nullptr;
  }

  


  void MatrixIO::readAndUpdateMatrix(std::string filename, Matrix* A)
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {
      A->setExpanded(false);
      std::string line;
      Int i = 0;
      Int m, n, nnz;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        //  std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m >> nnz;
      if ((A->getNumRows() != n) || (A->getNumColumns() != m) || (A->getNnz() < nnz)){      
        printf("Wrong matrix size! Cannot update \n ");
        exit(0);
      }
      A->setNnz(nnz);
      //create coo arrays
      Int* coo_rows = A->getCooRowIndices("cpu");
      Int* coo_cols = A->getCooColIndices("cpu");
      Real* coo_vals = A->getCooValues("cpu");
      i = 0;
      Int a, b;
      Real c;
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

  Real* MatrixIO::readAndUpdateRhs(std::string filename, Real* rhs) 
  {

    std::ifstream file(filename);
    std::stringstream ss;
    if (file.is_open()) {

      std::string line;
      Int i = 0;
      Int n, m;
      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m ;

      if(rhs == nullptr) { 
        Real* rhs = new Real[n];
      } 
      Real a;
      while (file >> a){
        rhs[i] = a;
        i++;
      }
      file.close();
      return rhs;
    }
    std::cout<<"Error opening file"<<std::endl;
    return nullptr;
  }

  Int MatrixIO::writeMatrixToFile(Matrix* A, std::string filename)
  {
    std::cout << "writeMatrixToFile function not implemented!\n";
    return -1;
  }

  Int MatrixIO::writeVectorToFile(Vector* vec_x, std::string filename)
  {
    Real* x_data = vec_x->getData("cpu");
    std::ofstream file_out (filename, std::ofstream::out);
    file_out << "%%MatrixMarket matrix array real general \n";
    file_out << "% ID: XXX \n";
    file_out << vec_x->getSize()<<" "<<1<<"\n";
    for (int i = 0; i < vec_x->getSize(); ++i) {
      file_out << std::setprecision(32)<<std::scientific << x_data[i]<<"\n";
    }
    file_out.close();
    return 0;
  }
}


