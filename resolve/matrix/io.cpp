#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <list>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Utilities.hpp>
#include "io.hpp"


namespace ReSolve { namespace io {

  static int loadToList(std::istream& file, std::list<CooTriplet>& tmp, bool is_expand_symmetric)
  {
    index_type idx = 0;
    index_type i, j;
    real_type v;
    while (file >> i >> j >> v) {
      CooTriplet triplet(i - 1, j - 1, v);
      tmp.push_back(std::move(triplet));
      idx++;
    }

    return 0;
  }

  static int removeDuplicates(std::list<CooTriplet>& tmp)
  {
    std::list<CooTriplet>::iterator it = tmp.begin();
    while (it != tmp.end())
    {
      std::list<CooTriplet>::iterator it_tmp = it;
      it++;
      if (*it == *it_tmp) {
        *it += *it_tmp;
        tmp.erase(it_tmp);
      }
    }

    return 0;
  }

  static int copyListToCoo(const std::list<CooTriplet>& tmp, matrix::Coo* A)
  {
    index_type* coo_rows = A->getRowData(memory::HOST);
    index_type* coo_cols = A->getColData(memory::HOST);
    real_type*  coo_vals = A->getValues( memory::HOST);

    index_type element_counter = 0;
    std::list<CooTriplet>::const_iterator it = tmp.begin();
    while (it != tmp.end())
    {
      coo_rows[element_counter] = it->getRowIdx();
      coo_cols[element_counter] = it->getColIdx();
      coo_vals[element_counter] = it->getValue();
      it++;
      element_counter++;
    }

    return 0;
  }

  matrix::Coo* readMatrixFromFile(std::istream& file, bool is_expand_symmetric)
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

    std::list<CooTriplet> tmp;

    // Store COO data in the temporary workspace.
    // Complexity O(NNZ)
    loadToList(file, tmp, is_expand_symmetric);

    // Sort tmp
    // Complexity O(NNZ*log(NNZ))
    tmp.sort();

    // Deduplicate tmp
    // Complexity O(NNZ)
    removeDuplicates(tmp);

    nnz = tmp.size();

    // Create matrix
    matrix::Coo* B = new matrix::Coo(n, m, nnz, symmetric, expanded);
    B->allocateMatrixData(memory::HOST);

    copyListToCoo(tmp, B);

    return B;
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

    // Default is not to expand symmetric matrix
    bool is_expand_symmetric = false;

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
    if ((A->getNumRows() != n) || (A->getNumColumns() != m)) {
      Logger::error() << "Wrong matrix size: " << A->getNumRows()
                      << "x" << A->getNumColumns()
                      << ". Cannot update! \n ";
      return;
    }

    std::list<CooTriplet> tmp;

    // Store COO data in the temporary workspace.
    // Complexity O(NNZ)
    loadToList(file, tmp, is_expand_symmetric);

    // Sort tmp
    // Complexity O(NNZ*log(NNZ))
    tmp.sort();

    // Deduplicate tmp
    // Complexity O(NNZ)
    removeDuplicates(tmp);

    // Set correct nnz after duplicates are merged. 
    nnz = tmp.size();
    if (A->getNnz() < nnz) {
      Logger::error() << "Too many NNZs: " << A->getNnz()
                      << ". Cannot update! \n ";
      return;
    }
    A->setNnz(nnz);

    // Populate COO matrix
    // Complexity O(NNZ)
    copyListToCoo(tmp, A);
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
