#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <list>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Utilities.hpp>
#include "io.hpp"


namespace ReSolve
{ 
  
  /**
   * @brief Helper class for COO matrix sorting.
   * 
   * The entire code is in this file. Its scope is to support matrix file I/O
   * only.
   * 
   */
  class CooTriplet
  {
    public:
      CooTriplet() : rowidx_(0), colidx_(0), value_(0.0)
      {}
      CooTriplet(index_type i, index_type j, real_type v) : rowidx_(i), colidx_(j), value_(v)
      {}
      ~CooTriplet()
      {}
      void setColIdx (index_type new_idx)
      {
        colidx_ = new_idx;
      }
      void setValue (real_type new_value)
      {
        value_ = new_value;
      }
      void set(index_type rowidx, index_type colidx, real_type value)
      {
        rowidx_ = rowidx;
        colidx_ = colidx;
        value_  = value;
      }

      index_type getRowIdx() const
      {
        return rowidx_;
      }
      index_type getColIdx() const
      {
        return colidx_;
      }
      real_type getValue() const
      {
        return value_;
      }

      bool operator < (const CooTriplet& str) const
      {
        if (rowidx_ < str.rowidx_)
          return true;

        if ((rowidx_ == str.rowidx_) && (colidx_ < str.colidx_))
          return true;

        return false;
      }

      bool operator == (const CooTriplet& str) const
      {
        return (rowidx_ == str.rowidx_) && (colidx_ == str.colidx_);
      }

      CooTriplet& operator += (const CooTriplet t)
      {
        if ((rowidx_ != t.rowidx_) || (colidx_ != t.colidx_)) {
          io::Logger::error() << "Adding values into non-matching triplet.\n";
        }
        value_ += t.value_;
        return *this;
      }

      void print() const
      {
        // Add 1 to indices to restore indexing from MM format
        std::cout << rowidx_ << " " << colidx_ << " " << value_ << "\n";
      }

    private:
      index_type rowidx_{0};
      index_type colidx_{0};
      real_type value_{0.0};
  };


  namespace io
  {

    // Static helper functionsdeclarations
    static void readListFromFile(std::istream& file,
                                 bool is_expand_symmetric,                                  
                                 std::list<CooTriplet>& tmp,
                                 index_type& n,
                                 index_type& m,
                                 index_type& nnz,
                                 bool& symmetric,
                                 bool& expanded);
    static void readAndUpdateSparseMatrix(std::istream& file,
                                          matrix::Sparse* A,
                                          std::list<CooTriplet>& tmp);
    // static void print_list(std::list<CooTriplet>& l);
    static int loadToList(std::istream& file, std::list<CooTriplet>& tmp, bool is_expand_symmetric);
    static int removeDuplicates(std::list<CooTriplet>& tmp);
    static int copyListToCoo(const std::list<CooTriplet>& tmp, matrix::Coo* A);
    static int copyListToCsr(const std::list<CooTriplet>& tmp, matrix::Csr* A);


    /**
     * @brief Create a COO matrix and populate it with data from Matrix Market
     * file.
     * 
     * @param file 
     * @param is_expand_symmetric 
     * @return matrix::Coo* 
     */
    matrix::Coo* readMatrixFromFile(std::istream& file, bool is_expand_symmetric)
    {
      if(!file) {
        Logger::error() << "Empty input to readMatrixFromFile function ... \n" << std::endl;
        return nullptr;
      }

      index_type m = 0, n = 0, nnz = 0;
      bool symmetric = false;
      bool expanded = true;

      std::list<CooTriplet> tmp;

      readListFromFile(file, is_expand_symmetric, tmp, n, m, nnz, symmetric, expanded);

      // Create matrix
      matrix::Coo* B = new matrix::Coo(n, m, nnz, symmetric, expanded);
      B->allocateMatrixData(memory::HOST);

      copyListToCoo(tmp, B);

      return B;
    }

    matrix::Csr* readCsrMatrixFromFile(std::istream& file, bool is_expand_symmetric)
    {
      if(!file) {
        Logger::error() << "Empty input to readMatrixFromFile function ... \n" << std::endl;
        return nullptr;
      }

      index_type m = 0, n = 0, nnz = 0;
      bool symmetric = false;
      bool expanded = true;

      std::list<CooTriplet> tmp;

      readListFromFile(file, is_expand_symmetric, tmp, n, m, nnz, symmetric, expanded);

      // Create matrix
      matrix::Csr* A = new matrix::Csr(n, m, nnz, symmetric, expanded);
      A->allocateMatrixData(memory::HOST);

      copyListToCsr(tmp, A);

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

      std::list<CooTriplet> tmp;

      readAndUpdateSparseMatrix(file, A, tmp);

      // Populate COO matrix. Complexity O(NNZ)
      copyListToCoo(tmp, A);
    }

    void readAndUpdateMatrix(std::istream& file, matrix::Csr* A)
    {
      if(!file) {
        Logger::error() << "Empty input to readMatrixFromFile function ..." << std::endl;
        return;
      }

      std::list<CooTriplet> tmp;

      readAndUpdateSparseMatrix(file, A, tmp);

      // Populate COO matrix. Complexity O(NNZ)
      copyListToCsr(tmp, A);
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


    //
    // Static helper functions
    //

    static void readListFromFile(std::istream& file,
                                 bool is_expand_symmetric,
                                 std::list<CooTriplet>& tmp,
                                 index_type& n,
                                 index_type& m,
                                 index_type& nnz,
                                 bool& symmetric,
                                 bool& expanded)
    {
      std::stringstream ss;
      std::string line;
      m = 0;
      n = 0;
      nnz = 0;
      symmetric = false;
      expanded = true;

      // Parse header and check if matrix is symmetric
      while (std::getline(file, line))
      {
        if (line.at(0) != '%')
          break;
        if (line.find("symmetric") != std::string::npos) {
          symmetric = true;
          expanded  = is_expand_symmetric;
        }
      }

      // Read the first line with matrix sizes
      ss << line;
      ss >> n >> m >> nnz;

      // Store COO data in the temporary workspace. Complexity O(NNZ)
      loadToList(file, tmp, symmetric && is_expand_symmetric);

      // Sort tmp. Complexity O(NNZ*log(NNZ))
      tmp.sort();

      // Deduplicate tmp. Complexity O(NNZ)
      removeDuplicates(tmp);

      nnz = static_cast<index_type>(tmp.size());
    }

    static void readAndUpdateSparseMatrix(std::istream& file, matrix::Sparse* A, std::list<CooTriplet>& tmp)
    {
      std::stringstream ss;
      std::string line;
      // Default is a general matrix
      bool symmetric = false;

      // Default is not to expand symmetric matrix
      bool is_expand_symmetric = false;

      // Parse header and check if matrix is symmetric
      std::getline(file, line);
      if (line.find("symmetric") != std::string::npos) {
        symmetric = true;
      }
      if (symmetric != A->symmetric()) {
        Logger::error() << "In function readAndUpdateMatrix:"
                        << "Source data does not match the symmetry of destination matrix.\n";
      }
      // If the destination matrix is symmetric and expanded, then expand data.
      if (A->symmetric()) {
        is_expand_symmetric = A->expanded();
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

      // Store COO data in the temporary workspace. Complexity O(NNZ)
      loadToList(file, tmp, is_expand_symmetric);

      // Sort tmp. Complexity O(NNZ*log(NNZ))
      tmp.sort();

      // Deduplicate tmp. Complexity O(NNZ)
      removeDuplicates(tmp);

      nnz = static_cast<index_type>(tmp.size());
      if (A->getNnz() < nnz) {
        Logger::error() << "Too many NNZs: " << A->getNnz()
                        << ". Cannot update! \n ";
        return;
      }
      A->setNnz(nnz);
    }


    // Commented out; needed for debugging only.
    // void print_list(std::list<CooTriplet>& l)
    // {
    //   // Print out the list
    //   std::cout << "tmp list:\n";
    //   for (CooTriplet& n : l)
    //     n.print();
    //   std::cout << "\n";
    // }

    int loadToList(std::istream& file, std::list<CooTriplet>& tmp, bool is_expand_symmetric)
    {
      index_type i, j;
      real_type v;
      while (file >> i >> j >> v) {
        CooTriplet triplet(i - 1, j - 1, v);
        tmp.push_back(std::move(triplet));
        if (is_expand_symmetric) {
          if (i != j) {
            CooTriplet triplet(j - 1, i - 1, v);
            tmp.push_back(std::move(triplet));
          }
        }
      }

      return 0;
    }

    int removeDuplicates(std::list<CooTriplet>& tmp)
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

    int copyListToCoo(const std::list<CooTriplet>& tmp, matrix::Coo* A)
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

    int copyListToCsr(const std::list<CooTriplet>& tmp, matrix::Csr* A)
    {
      index_type* csr_rows = A->getRowData(memory::HOST);
      index_type* csr_cols = A->getColData(memory::HOST);
      real_type*  csr_vals = A->getValues( memory::HOST);

      // Set number of nonzeros
      index_type nnz = static_cast<index_type>(tmp.size());

      // Set all iterators
      index_type column_index_counter = 0;
      index_type row_pointer_counter = 0;
      std::list<CooTriplet>::const_iterator it = tmp.begin();

      // Set first row pointer to zero
      csr_rows[0] = 0;
      csr_cols[0] = it->getColIdx();
      csr_vals[0] = it->getValue();

      for (index_type i = 1; i < nnz; ++i) {
        std::list<CooTriplet>::const_iterator it_tmp = it;
        it++;
        if (it->getRowIdx() != it_tmp->getRowIdx()) {
          row_pointer_counter++;
          csr_rows[row_pointer_counter] = i;
        }
        column_index_counter++;
        csr_cols[column_index_counter] = it->getColIdx();
        csr_vals[column_index_counter] = it->getValue();
      }
      row_pointer_counter++;
      csr_rows[row_pointer_counter] = nnz;

      return 0;
    }

  } // namespace io
} // namespace ReSolve
