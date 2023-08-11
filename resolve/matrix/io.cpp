#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include "io.hpp"


namespace ReSolve { namespace io {

  matrix::Coo* readMatrixFromFile(std::istream& file)
  {
    if(!file) {
      std::cout << "Empty input to readMatrixFromFile function ... \n" <<std::endl;
      return nullptr;
    }

    std::stringstream ss;
    std::string line;
    index_type i = 0;
    index_type m, n, nnz;
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
    matrix::Coo* A = new matrix::Coo(n, m, nnz,symmetric, expanded );  
    //create coo arrays
    index_type* coo_rows = new index_type[nnz];
    index_type* coo_cols = new index_type[nnz];
    real_type* coo_vals = new real_type[nnz];
    i = 0;
    index_type a, b;
    real_type c;
    while (file >> a >> b >> c) {
      coo_rows[i] = a - 1;
      coo_cols[i] = b - 1;
      coo_vals[i] = c;
      i++;
    }
    A->setMatrixData(coo_rows, coo_cols, coo_vals, "cpu");
    return A;
  }


  real_type* readRhsFromFile(std::istream& file)
  {
    if(!file) {
      std::cout << "Empty input to " << __func__ << " function ... \n" << std::endl;
      return nullptr;
    }

    std::stringstream ss;
    std::string line;
    index_type i = 0;
    index_type n, m;

    std::getline(file, line);
    while (line.at(0) == '%') {
      std::getline(file, line); 
      // std::cout << line << std::endl;
    }
    ss << line;
    ss >> n >> m ;

    real_type* vec = new real_type[n];
    real_type a;
    while (file >> a){
      vec[i] = a;
      i++;
    }
    return vec;
  }

  void readAndUpdateMatrix(std::istream& file, matrix::Coo* A)
  {
    if(!file) {
      std::cout << "Empty input to readMatrixFromFile function ... \n" <<std::endl;
      return;
    }

    std::stringstream ss;
    A->setExpanded(false);
    std::string line;
    index_type i = 0;
    index_type m, n, nnz;
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
    index_type* coo_rows = A->getRowData("cpu");
    index_type* coo_cols = A->getColData("cpu");
    real_type* coo_vals = A->getValues("cpu");
    i = 0;
    index_type a, b;
    real_type c;
    while (file >> a >> b >> c) {
      coo_rows[i] = a - 1;
      coo_cols[i] = b - 1;
      coo_vals[i] = c;
      i++;
    }
  }

  void readAndUpdateRhs(std::istream& file, real_type** p_rhs) 
  {
    if (!file) {
      std::cout << "Empty input to readAndUpdateRhs function ... \n" <<std::endl;
      return;
    }

    real_type* rhs = *p_rhs;
    std::stringstream ss;
    std::string line;
    index_type n, m;

    std::getline(file, line);
    while (line.at(0) == '%') {
      std::getline(file, line); 
      // std::cout<<line<<std::endl;
    }
    ss << line;
    ss >> n >> m ;

    if (rhs == nullptr) {
      // std::cout << "Allocating array of size " << n << "\n";
      rhs = new real_type[n];
    } 
    real_type a;
    index_type i = 0;
    while (file >> a) {
      rhs[i] = a;
      // std::cout << i << ": " << a << "\n";
      i++;
    }
  }

  int writeMatrixToFile(matrix::Sparse* A, std::ostream& file_out)
  {
    std::cout << "writeMatrixToFile function not implemented!\n";
    return -1;
  }

  int writeVectorToFile(vector_type* vec_x, std::ostream& file_out)
  {
    real_type* x_data = vec_x->getData("cpu");
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

}} // ReSolve::io
