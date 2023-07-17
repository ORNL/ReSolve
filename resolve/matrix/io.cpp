#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include <resolve/MatrixCOO.hpp>
#include "io.hpp"


namespace ReSolve { namespace matrix { namespace io {

  MatrixCOO* readMatrixFromFile(std::istream& file)
  {
    if(!file) {
      std::cout << "Empty input to readMatrixFromFile function ... \n" <<std::endl;
      return nullptr;
    }

    std::stringstream ss;
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
    MatrixCOO* A = new MatrixCOO(n, m, nnz,symmetric, expanded );  
    //create coo arrays
    Int* coo_rows = new Int[nnz];
    Int* coo_cols = new Int[nnz];
    Real* coo_vals = new Real[nnz];
    i = 0;
    Int a, b;
    Real c;
    while (file >> a >> b >> c) {
      coo_rows[i] = a - 1;
      coo_cols[i] = b - 1;
      coo_vals[i] = c;
      i++;
    }
    A->setMatrixData(coo_rows, coo_cols, coo_vals, "cpu");
    return A;
  }


  Real* readRhsFromFile(std::istream& file)
  {
    if(!file) {
      std::cout << "Empty input to " << __func__ << " function ... \n" << std::endl;
      return nullptr;
    }

    std::stringstream ss;
    std::string line;
    Int i = 0;
    Int n, m;

    std::getline(file, line);
    while (line.at(0) == '%') {
      std::getline(file, line); 
      // std::cout << line << std::endl;
    }
    ss << line;
    ss >> n >> m ;

    Real* vec = new Real[n];
    Real a;
    while (file >> a){
      vec[i] = a;
      i++;
    }
    return vec;
  }

  void readAndUpdateMatrix(std::istream& file, MatrixCOO* A)
  {
    if(!file) {
      std::cout << "Empty input to readMatrixFromFile function ... \n" <<std::endl;
      return;
    }

    std::stringstream ss;
    A->setExpanded(false);
    std::string line;
    Int i = 0;
    Int m, n, nnz;
    std::getline(file, line);
    while (line.at(0) == '%') {
      std::getline(file, line); 
      //  std::cout << line << std::endl;
    }

    ss << line;
    ss >> n >> m >> nnz;
    if ((A->getNumRows() != n) || (A->getNumColumns() != m) || (A->getNnz() < nnz)) {      
      std::cout << "Wrong matrix size: " << A->getNumRows() << "x" << A->getNumColumns() 
                << ", NNZ: " << A->getNnz() << " Cannot update! \n ";
      exit(0);
    }
    A->setNnz(nnz);
    //create coo arrays
    Int* coo_rows = A->getRowData("cpu");
    Int* coo_cols = A->getColData("cpu");
    Real* coo_vals = A->getValues("cpu");
    i = 0;
    Int a, b;
    Real c;
    while (file >> a >> b >> c) {
      coo_rows[i] = a - 1;
      coo_cols[i] = b - 1;
      coo_vals[i] = c;
      i++;
    }
  }

  void readAndUpdateRhs(std::istream& file, Real** p_rhs) 
  {
    if (!file) {
      std::cout << "Empty input to readAndUpdateRhs function ... \n" <<std::endl;
      return;
    }

    Real* rhs = *p_rhs;
    std::stringstream ss;
    std::string line;
    Int n, m;

    std::getline(file, line);
    while (line.at(0) == '%') {
      std::getline(file, line); 
      // std::cout<<line<<std::endl;
    }
    ss << line;
    ss >> n >> m ;

    if (rhs == nullptr) {
      // std::cout << "Allocating array of size " << n << "\n";
      rhs = new Real[n];
    } 
    Real a;
    Int i = 0;
    while (file >> a) {
      rhs[i] = a;
      // std::cout << i << ": " << a << "\n";
      i++;
    }
  }

  int writeMatrixToFile(Matrix* A, std::ostream file_out)
  {
    std::cout << "writeMatrixToFile function not implemented!\n";
    return -1;
  }

  int writeVectorToFile(Vector* vec_x, std::ostream file_out)
  {
    Real* x_data = vec_x->getData("cpu");
    // std::ofstream file_out (filename, std::ofstream::out);
    file_out << "%%MatrixMarket matrix array real general \n";
    file_out << "% ID: XXX \n";
    file_out << vec_x->getSize() << " " << 1 << "\n";
    for (int i = 0; i < vec_x->getSize(); ++i) {
      file_out << std::setprecision(32) << std::scientific << x_data[i] << "\n";
    }
    // file_out.close();
    return 0;
  }

}}} // ReSolve::matrix::io