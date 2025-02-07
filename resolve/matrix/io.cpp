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
#include "io.hpp"


namespace ReSolve
{ 
  
  /**
   * @class MatrixElementTriplet
   * 
   * @brief Helper class for COO matrix sorting.
   * 
   * Contains triplet of row index, column index and the value of a sparse
   * matrix element, as well as methods and operator overloads for its
   * management.
   * 
   * The entire code is in this file. Its scope is to support matrix file I/O
   * only.
   * 
   */
  class MatrixElementTriplet
  {
    public:
      /// Default constructor initializes all to zero.
      MatrixElementTriplet() : rowidx_(0), colidx_(0), value_(0.0)
      {}

      /// Constructor that initializes row and column indices and the element value.
      MatrixElementTriplet(index_type i, index_type j, real_type v) : rowidx_(i), colidx_(j), value_(v)
      {}

      ~MatrixElementTriplet() = default;

      /// Set the row and column indices and the element value.
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

      /**
       * @brief Overload of `<` operator
       * 
       * Ensures that matrix elements stored in MatrixElementTriplet will be
       * sorted by their indices in a row-major order.
       * 
       */
      bool operator < (const MatrixElementTriplet& t) const
      {
        if (rowidx_ < t.rowidx_)
          return true;

        if ((rowidx_ == t.rowidx_) && (colidx_ < t.colidx_))
          return true;

        return false;
      }

      /**
       * @brief Overload of `==` operator.
       * 
       * This overload is used to indicate when two different instances of
       * MatrixElementTriplet correspond to the same matrix element.
       */
      bool operator == (const MatrixElementTriplet& str) const
      {
        return (rowidx_ == str.rowidx_) && (colidx_ == str.colidx_);
      }

      /**
       * @brief Overload of `+=` operator.
       * 
       * @param t - Triplet to be added in place.
       * @return MatrixElementTriplet& - reference to `*this`.
       * 
       * This overload is used to merge duplicates in sparse matrix in COO
       * format. It will return error and leave `*this` unchanged if the
       * argument corresponds to an element with different row or column
       * indices.
       */
      MatrixElementTriplet& operator += (const MatrixElementTriplet t)
      {
        if ((rowidx_ != t.rowidx_) || (colidx_ != t.colidx_)) {
          io::Logger::error() << "Adding values into non-matching triplet.\n";
          return *this;
        }
        value_ += t.value_;
        return *this;
      }

      /// Utility to print indices (0 index base).
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
    static void createMatrixFromFileAsList(std::istream& file,
                                           bool is_expand_symmetric,                                  
                                           std::list<MatrixElementTriplet>& tmp,
                                           index_type& n,
                                           index_type& m,
                                           index_type& nnz,
                                           bool& symmetric,
                                           bool& expanded);
    static void createMatrixFromFileAsList(std::istream& file,
                                           matrix::Sparse* A,
                                           std::list<MatrixElementTriplet>& tmp);
    // static void print_list(std::list<MatrixElementTriplet>& l);
    static int loadToList(std::istream& file, bool is_expand_symmetric, std::list<MatrixElementTriplet>& tmp);
    static int removeDuplicates(std::list<MatrixElementTriplet>& tmp);
    static int copyListToCoo(const std::list<MatrixElementTriplet>& tmp, matrix::Coo* A);
    static int copyListToCsr(const std::list<MatrixElementTriplet>& tmp, matrix::Csr* A);


    /**
     * @brief Create a COO matrix and populate it with data from Matrix Market
     * file.
     * 
     * @param file - input Matrix Market file
     * @param is_expand_symmetric - whether to expand symmetric matrix to general format
     * @return matrix::Coo* - pointer to COO matrix
     * 
     * @pre file is a valid std::istream with Matrix Market data.
     * @pre input data is in valid Matrix Market format.
     * @post Valid COO matrix sorted in row major order and without duplicates
     * is created.
     */
    matrix::Coo* createCooFromFile(std::istream& file, bool is_expand_symmetric)
    {
      if(!file) {
        Logger::error() << "Empty input to createCooFromFile function ... \n" << std::endl;
        return nullptr;
      }

      index_type m = 0, n = 0, nnz = 0;
      bool symmetric = false;
      bool expanded = true;

      std::list<MatrixElementTriplet> tmp;

      createMatrixFromFileAsList(file, is_expand_symmetric, tmp, n, m, nnz, symmetric, expanded);

      // Create matrix
      matrix::Coo* B = new matrix::Coo(n, m, nnz, symmetric, expanded);
      B->allocateMatrixData(memory::HOST);

      copyListToCoo(tmp, B);

      return B;
    }

    /**
     * @brief 
     * 
     * @param file - input Matrix Market file
     * @param is_expand_symmetric - whether to expand symmetric matrix to general format
     * @return matrix::Csr* - pointer to COO matrix
     * 
     * @pre file is a valid std::istream with Matrix Market data.
     * @pre input data is in valid Matrix Market format.
     * @post Valid CSR matrix sorted in row major order and without duplicates
     * is created.
     */
    matrix::Csr* createCsrFromFile(std::istream& file, bool is_expand_symmetric)
    {
      if(!file) {
        Logger::error() << "Empty input to createCooFromFile function ... \n" << std::endl;
        return nullptr;
      }

      index_type m = 0, n = 0, nnz = 0;
      bool symmetric = false;
      bool expanded = true;

      std::list<MatrixElementTriplet> tmp;

      createMatrixFromFileAsList(file, is_expand_symmetric, tmp, n, m, nnz, symmetric, expanded);

      // Create matrix
      matrix::Csr* A = new matrix::Csr(n, m, nnz, symmetric, expanded);
      A->allocateMatrixData(memory::HOST);

      copyListToCsr(tmp, A);

      return A;
    }

    /**
     * @brief Imports vector data from a Matrix Market file.
     * 
     * @param file - std::istream to Matrix Market file (data).
     * @return real_type* - pointer to array with (dense) vector entries.
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @post A raw array with vector data is created.
     * 
     */
    real_type* createArrayFromFile(std::istream& file)
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
      }
      ss << line;
      ss >> n >> m;

      real_type* vec = new real_type[n];
      real_type a;
      while (file >> a) {
        vec[i] = a;
        i++;
      }
      return vec;
    }

    vector::Vector* createVectorFromFile(std::istream& file)
    {
      if(!file) {
        Logger::error() << "Empty input to " << __func__ << " function ... \n";
        return nullptr;
      }

      std::stringstream ss;
      std::string line;
      index_type i = 0;
      index_type n, m;

      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
      }
      ss << line;
      ss >> n >> m;

      vector::Vector* vec = new vector::Vector(n);
      vec->allocate(memory::HOST);
      real_type a;
      while (file >> a) {
        vec->getData(memory::HOST)[i] = a;
        i++;
      }
      vec->setDataUpdated(memory::HOST);
      return vec;
    }

    /**
     * @brief Reads data from a Matrix Market file and updates COO matrix A.
     * 
     * Compute complexity of this function is O(NNZ*log(NNZ)). There is an
     * overload of this function that generates a CSR matrix.
     * 
     * @param file - std::istream to Matrix Market file (data).
     * @param A - output COO matrix.
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @pre Size of matrix stored in the Matrix Market file matches the size of A.
     * @post Valid COO matrix sorted in row major order and without duplicates
     * is created.
     */
    void updateMatrixFromFile(std::istream& file, matrix::Coo* A)
    {
      if (!file) {
        Logger::error() << "Empty input to createCooFromFile function ..." << std::endl;
        return;
      }

      std::list<MatrixElementTriplet> tmp;

      createMatrixFromFileAsList(file, A, tmp);

      // Populate COO matrix. Complexity O(NNZ)
      copyListToCoo(tmp, A);
    }

    /**
     * @brief Reads data from a Matrix Market file and updates CSR matrix A.
     * 
     * Compute complexity of this function is O(NNZ*log(NNZ)). There is an
     * overload of this function that generates a COO matrix.
     * 
     * @param file - std::istream to Matrix Market file (data).
     * @param A - output CSR matrix.
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @pre Size of matrix stored in the Matrix Market file matches the size of A.
     * @post Valid CSR matrix sorted in row major order and without duplicates
     * is created.
     */
    void updateMatrixFromFile(std::istream& file, matrix::Csr* A)
    {
      if(!file) {
        Logger::error() << "Empty input to updateMatrixFromFile function ..." << std::endl;
        return;
      }

      std::list<MatrixElementTriplet> tmp;

      createMatrixFromFileAsList(file, A, tmp);

      // Populate COO matrix. Complexity O(NNZ)
      copyListToCsr(tmp, A);
    }

    /**
     * @brief Reads data from a Matrix Market file and updates array p_rhs.
     * 
     * @param file - std::istream to Matrix Market file (data).
     * @param p_rhs - pointer to a pointer to a raw array with vector data.
     * 
     * @todo The righ-hand-side should be of vector type, not a raw array. With
     * current implementation it is impossible to verify if the sufficient
     * space is allocated to store all the data from the input file. Risk of
     * writing past the end of the array is high.
     */
    void updateArrayFromFile(std::istream& file, real_type** p_rhs) 
    {
      if (!file) {
        Logger::error() << "Empty input to updateArrayFromFile function ..." << std::endl;
        return;
      }

      real_type* rhs = *p_rhs;
      std::stringstream ss;
      std::string line;
      index_type n, m;

      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
      }
      ss << line;
      ss >> n >> m;

      if (rhs == nullptr) {
        rhs = new real_type[n];
      } 
      real_type a;
      index_type i = 0;
      while (file >> a) {
        rhs[i] = a;
        i++;
      }
    }

    void updateVectorFromFile(std::istream& file, vector::Vector* vec_rhs) 
    {
      if (!file) {
        Logger::error() << "Empty input to updateArrayFromFile function ..." << std::endl;
        return;
      }

      std::stringstream ss;
      std::string line;
      index_type n, m;

      std::getline(file, line);
      while (line.at(0) == '%') {
        std::getline(file, line); 
        // std::cout<<line<<std::endl;
      }
      ss << line;
      ss >> n >> m;

      if (n != vec_rhs->getSize()) {
        Logger::error() << "File data does not match the vector size.\n"
                        << "Vector not updated!\n";
        return;
      }

      real_type* rhs = vec_rhs->getData(memory::HOST);
      real_type a = 0.0;
      index_type i = 0;
      while (file >> a) {
        rhs[i] = a;
        // std::cout << i << ": " << a << "\n";
        i++;
      }
      vec_rhs->setDataUpdated(memory::HOST);
    }

    /**
     * @brief Writes matrix A to a file in Matrix Market format.
     * 
     * @param A - input matrix.
     * @param file_out - std::ostream to output file.
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `A` is a valid sparse matrix.
     * @post Valid Matrix Marked data is written to std::ostream.
     * @invariant Matrix `A` elements are unchanged.
     */
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
      
      index_type indexing_base = 1;
      A->print(file_out, indexing_base);
      return 0;
    }

    /**
     * @brief Writes vector data to a file in Matrix Market format.
     * 
     * @param vec_x - Input vector.
     * @param file_out - std::ostream to output file.
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `vec_x` is a valid vector.
     * @post Valid Matrix Market data is written to std::ostream.
     * @invariant Elements of `vec_x` are unchanged.
     */
    int writeVectorToFile(vector_type* vec_x, std::ostream& file_out)
    {
      real_type* x_data = vec_x->getData(memory::HOST);

      file_out << "%%MatrixMarket matrix array real general \n";
      file_out << "% Generated by Re::Solve <https://github.com/ORNL/ReSolve>\n";
      file_out << vec_x->getSize() << " " << 1 << "\n";
      for (int i = 0; i < vec_x->getSize(); ++i) {
        file_out << std::setprecision(std::numeric_limits<real_type>::digits10 + 1)
                 << std::scientific << x_data[i] << "\n";
      }

      return 0;
    }


    //
    // Static helper functions
    //

    /**
     * @brief Reads Matrix Market data from std::istream and stores it into
     * std::list<MatrixElementTriplet>.
     * 
     * @param[in]  file - std::istream to Matrix Market file (data).
     * @param[in]  is_expand_symmetric - whether to expand symmetric matrix to general format
     * @param[out] tmp - std::list where to store matrix data
     * @param[out] n - number of rows as read from Matrix Market file
     * @param[out] m - number of columns as read from Matrix Market file 
     * @param[out] nnz - calculated number of matrix nonzeros
     * @param[out] symmetric - if matrix is symmetric
     * @param[out] expanded - if symmetric matrix is expanded to general format
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @pre `tmp` is an empty list!
     * @post Matrix size is set to values read from the Matrix Market file.
     * @post `nnz` is number of nonzeros in the matrix after duplicates are
     * removed and the matrix is (optionally) expanded.
     * @post `symmetric` and `expanded` flags are set based on data read into
     * `tmp` list.
     * @post `tmp` list is overwritten with matrix elements read from the input
     * stream.
     */
    static void createMatrixFromFileAsList(std::istream& file,
                                           bool is_expand_symmetric,
                                           std::list<MatrixElementTriplet>& tmp,
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
      loadToList(file, symmetric && is_expand_symmetric, tmp);

      // Sort tmp. Complexity O(NNZ*log(NNZ)).
      tmp.sort();

      // Deduplicate tmp. Complexity O(NNZ).
      removeDuplicates(tmp);

      // Set nnz without duplicates and for possibly expanded matrix.
      nnz = static_cast<index_type>(tmp.size());
    }

    /**
     * @brief 
     * 
     * @param[in]  file - std::istream to Matrix Market file (data).
     * @param[in]  A - sparse matrix to be updated
     * @param[out] tmp - temporary list with matrix entries
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @pre `tmp` is an empty list!
     * @pre Matrix size of `A` must match matrix size read from the Matrix Market
     * file.
     * @pre Number of nonzeros in `A` must not be smaller than the number of
     * zeros read from the Matrix Market file.
     * @post `tmp` list is overwritten with matrix elements read from the input
     * stream.
     * @invariant Elements of `A` are unchanged in this function but they are
     * expected to be overwritten with values in `tmp` later in the code. 
     */
    static void createMatrixFromFileAsList(std::istream& file,
                                           matrix::Sparse* A,
                                           std::list<MatrixElementTriplet>& tmp)
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
        Logger::error() << "In function updateMatrixFromFile:"
                        << "Source data does not match the symmetry of destination matrix.\n";
      }
      // If the destination matrix is symmetric and expanded, then expand data.
      if (A->symmetric()) {
        is_expand_symmetric = A->expanded();
      }

      // Skip the header comments
      while (line.at(0) == '%') {
        std::getline(file, line); 
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
      loadToList(file, is_expand_symmetric, tmp);

      // Sort tmp. Complexity O(NNZ*log(NNZ))
      tmp.sort();

      // Remove duplicates in `tmp` list. Complexity O(NNZ)
      removeDuplicates(tmp);

    }


    // Commented out; needed for debugging only.
    // void print_list(std::list<MatrixElementTriplet>& l)
    // {
    //   // Print out the list
    //   std::cout << "tmp list:\n";
    //   for (MatrixElementTriplet& n : l)
    //     n.print();
    //   std::cout << "\n";
    // }

    /**
     * @brief Loads data from Matrix Market file to a std::list.
     * 
     * @param[in]  file - std::istream to Matrix Market file (data).
     * @param[in]  is_expand_symmetric - whether to expand symmetric matrix.
     * @param[out] tmp - temporary list with matrix entries
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `file` is a valid std::istream with Matrix Market data.
     * @pre Input data is in valid Matrix Market format.
     * @pre `tmp` is an empty list!
     * @post `tmp` list is overwritten with matrix elements read from the input
     * stream.
     */
    int loadToList(std::istream& file,
                   bool is_expand_symmetric,
                   std::list<MatrixElementTriplet>& tmp)
    {
      index_type i, j;
      real_type v;

      // If the `tmp` list is not empty, clear it.
      if (tmp.size() != 0) {
        tmp.clear();
      }

      while (file >> i >> j >> v) {
        MatrixElementTriplet triplet(i - 1, j - 1, v);
        tmp.push_back(std::move(triplet));
        if (is_expand_symmetric) {
          if (i != j) {
            MatrixElementTriplet triplet(j - 1, i - 1, v);
            tmp.push_back(std::move(triplet));
          }
        }
      }

      return 0;
    }

    /**
     * @brief Removes duplicates from `tmp` list.
     * 
     * @param[in,out] tmp - List with matrix entries. 
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `tmp` contains matrix elements.
     * @post Duplicates in `tmp` are added in place to the first instance
     * of that matrix element.
     */
    int removeDuplicates(std::list<MatrixElementTriplet>& tmp)
    {
      std::list<MatrixElementTriplet>::iterator it = tmp.begin();
      while (it != tmp.end())
      {
        std::list<MatrixElementTriplet>::iterator it_tmp = it;
        it++;
        if (*it == *it_tmp) {
          *it += *it_tmp;
          tmp.erase(it_tmp);
        }
      }

      return 0;
    }

    /**
     * @brief Writes data from the std::list to COO matrix.
     * 
     * @param[in]  tmp - List with matrix entries
     * @param[out] A   - Output COO matrix
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `tmp` contains matrix elements sorted in row-major order and
     * without duplicates.
     * @pre Number of `tmp` elements is not larger than number of nonzeros
     * in `A`.
     * @post Matrix data in `A` is updated with data from `tmp`.
     */
    int copyListToCoo(const std::list<MatrixElementTriplet>& tmp, matrix::Coo* A)
    {
      index_type* coo_rows = A->getRowData(memory::HOST);
      index_type* coo_cols = A->getColData(memory::HOST);
      real_type*  coo_vals = A->getValues( memory::HOST);

      index_type nnz = static_cast<index_type>(tmp.size());
      if (A->getNnz() < nnz) {
        Logger::error() << "Too many NNZs: " << nnz
                        << ". Cannot update! \n ";
        return 1;
      }
      A->setNnz(nnz);

      index_type element_counter = 0;
      std::list<MatrixElementTriplet>::const_iterator it = tmp.begin();
      while (it != tmp.end())
      {
        coo_rows[element_counter] = it->getRowIdx();
        coo_cols[element_counter] = it->getColIdx();
        coo_vals[element_counter] = it->getValue();
        it++;
        element_counter++;
      }

      // We updated matrix values outside Matrix API. We need to note that.
      A->setUpdated(memory::HOST);

      return 0;
    }


    /**
     * @brief Writes data from the std::list to CSR matrix.
     * 
     * @param[in]  tmp - List with matrix entries
     * @param[out] A   - Output CSR matrix
     * @return int - 0 if successful, error code otherwise.
     * 
     * @pre `tmp` contains matrix elements sorted in row-major order and
     * without duplicates.
     * @pre Number of `tmp` elements is not larger than number of nonzeros
     * in `A`.
     * @post Matrix data in `A` is updated with data from `tmp`.
     */
    int copyListToCsr(const std::list<MatrixElementTriplet>& tmp, matrix::Csr* A)
    {
      index_type* csr_rows = A->getRowData(memory::HOST);
      index_type* csr_cols = A->getColData(memory::HOST);
      real_type*  csr_vals = A->getValues( memory::HOST);

      // Set number of nonzeros
      index_type nnz = static_cast<index_type>(tmp.size());
      if (A->getNnz() < nnz) {
        Logger::error() << "Too many NNZs: " << nnz
                        << ". Cannot update! \n ";
        return 1;
      }
      A->setNnz(nnz);

      // Set all iterators
      index_type column_index_counter = 0;
      index_type row_pointer_counter = 0;
      std::list<MatrixElementTriplet>::const_iterator it = tmp.begin();

      // Set first row pointer to zero
      csr_rows[0] = 0;
      csr_cols[0] = it->getColIdx();
      csr_vals[0] = it->getValue();

      for (index_type i = 1; i < nnz; ++i) {
        std::list<MatrixElementTriplet>::const_iterator it_tmp = it;
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

      // We updated matrix values outside Matrix API. We need to note that.
      A->setUpdated(memory::HOST);

      return 0;
    }

  } // namespace io
} // namespace ReSolve
