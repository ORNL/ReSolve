#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include "io.hpp"


namespace ReSolve { namespace io {

  matrix::Coo* readMatrixFromFile(std::istream& file)
  {
    if(!file) {
      Logger::error() << "Empty input to readMatrixFromFile function ... \n" << std::endl;
      return nullptr;
    }

    std::stringstream ss;
    std::string line;
    index_type m, n, nnz;
    bool symmetric = false;
    bool expanded = true;

    // Parse header and check if matrix is symmetric
    while (std::getline(file, line))
    {
      if (line.at(0) != '%')
        break;
      if (line.find("symmetric") != std::string::npos) {
        symmetric = true;
        expanded = false;
      }
    }

    // Read the first line with matrix sizes
    ss << line;
    ss >> n >> m >> nnz;

    // Allocate and COO matrix data arrays
    index_type* coo_rows = new index_type[nnz];
    index_type* coo_cols = new index_type[nnz];
    real_type* coo_vals = new real_type[nnz];

    // Set COO data arrays
    index_type idx = 0;
    index_type i, j;
    real_type v;
    while (file >> i >> j >> v) {
      coo_rows[idx] = i - 1;
      coo_cols[idx] = j - 1;
      coo_vals[idx] = v;
      idx++;
    }

    // Create matrix object
    matrix::Coo* A = new matrix::Coo(n,
                                     m,
                                     nnz,
                                     symmetric,
                                     expanded,
                                     &coo_rows,
                                     &coo_cols,
                                     &coo_vals,
                                     memory::HOST,
                                     memory::HOST);  
    return A;
  }


  real_type* readRhsFromFile(std::istream& file)
  {
    if(!file) {
      Logger::error() << "Empty input to " << __func__ << " function ... \n" << std::endl;
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
      Logger::error() << "Empty input to readMatrixFromFile function ..." << std::endl;
      return;
    }

    std::stringstream ss;
    std::string line;
    // Default is a general matrix
    bool symmetric = false;
    bool expanded = true;

    // Parse header and check if matrix is symmetric
    std::getline(file, line);
    if (line.find("symmetric") != std::string::npos) {
      symmetric = true;
      expanded = false;
    }
    if (symmetric != A->symmetric()) {
      Logger::error() << "In function readAndUpdateMatrix:"
                      << "Source data does not match the symmetry of destination matrix.\n";
    }
    if (A->symmetric()) {
      if (expanded != A->expanded()) {
        Logger::error() << "In function readAndUpdateMatrix:"
                        << "Source data symmetric but the destination matrix is expanded.\n";
      }
    }

    // Skip the header comments
    while (line.at(0) == '%') {
      std::getline(file, line); 
      //  std::cout << line << std::endl;
    }

    // Read the first line with matrix sizes
    index_type m, n, nnz;
    ss << line;
    ss >> n >> m >> nnz;

    // Make sure input data matches matrix A size
    if ((A->getNumRows() != n) || (A->getNumColumns() != m) || (A->getNnz() < nnz)) {      
      Logger::error() << "Wrong matrix size: " << A->getNumRows()
                      << "x" << A->getNumColumns() 
                      << ", NNZ: " << A->getNnz()
                      << " Cannot update! \n ";
      return;
    }
    A->setNnz(nnz);

    // Populate COO data arrays
    index_type* coo_rows = A->getRowData(memory::HOST);
    index_type* coo_cols = A->getColData(memory::HOST);
    real_type* coo_vals  = A->getValues( memory::HOST);
    index_type idx = 0;
    index_type i, j;
    real_type v;
    while (file >> i >> j >> v) {
      coo_rows[idx] = i - 1;
      coo_cols[idx] = j - 1;
      coo_vals[idx] = v;
      idx++;
    }
  }

  void readAndUpdateRhs(std::istream& file, real_type** p_rhs) 
  {
    if (!file) {
      Logger::error() << "Empty input to readAndUpdateRhs function ..." << std::endl;
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
    if (A == nullptr) {
      Logger::error() << "Matrix pointer is NULL!\n";
      return -1;
    }

    if (A->symmetric() && !A->expanded()) {
      file_out << "%%MatrixMarket matrix coordinate real symmetric\n";
    } else {
      file_out << "%%MatrixMarket matrix coordinate real general\n";
    }
    file_out << "% Generated by Re::Solve <https://github.com/ORNL/ReSolve>\n";
    file_out << A->getNumRows()    << " " 
             << A->getNumColumns() << " "
             << A->getNnz()        << "\n";
    A->print(file_out, 1);
    // Indexing base 1 ^^
    return 0;
  }

  int writeVectorToFile(vector_type* vec_x, std::ostream& file_out)
  {
    real_type* x_data = vec_x->getData(memory::HOST);
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
