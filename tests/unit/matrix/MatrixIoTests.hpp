#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <tests/unit/TestBase.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve { namespace tests {

class MatrixIoTests : TestBase
{
public:
  MatrixIoTests(){}
  virtual ~MatrixIoTests(){}


  TestOutcome cooMatrixImport()
  {
    TestStatus status;

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file(general_coo_matrix_file_);
    ReSolve::matrix::Coo* A = ReSolve::io::createCooFromFile(file);

    // Check if the matrix data was correctly loaded
    status = true;

    index_type nnz_answer = static_cast<index_type>(general_coo_matrix_vals_.size());
    if (A->getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if (A->symmetric()) {
      std::cout << "Incorrect matrix type, matrix not symmetric ...\n";
      status = false;
    }

    if (!A->expanded()) {
      std::cout << "Incorrect matrix type, matrix is general (expanded) ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, general_coo_matrix_rows_, general_coo_matrix_cols_, general_coo_matrix_vals_);

    // A->print();
    delete A;
    A = nullptr;

    bool is_expand_symmetric = false;
    std::istringstream file2(symmetric_duplicates_coo_matrix_file_);
    A = ReSolve::io::createCooFromFile(file2, is_expand_symmetric);

    nnz_answer = static_cast<index_type>(symmetric_coo_matrix_vals_.size());
    if (A->getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if (!A->symmetric()) {
      std::cout << "Incorrect matrix type, matrix is symmetric ...\n";
      status = false;
    }

    if (A->expanded()) {
      std::cout << "Incorrect matrix type, matrix not expanded ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, symmetric_coo_matrix_rows_, symmetric_coo_matrix_cols_, symmetric_coo_matrix_vals_);

    delete A;
    A = nullptr;

    is_expand_symmetric = true;
    std::istringstream file3(symmetric_duplicates_coo_matrix_file_);
    A = ReSolve::io::createCooFromFile(file3, is_expand_symmetric);

    nnz_answer = static_cast<index_type>(symmetric_expanded_coo_matrix_vals_.size());
    if (A->getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if (!A->symmetric()) {
      std::cout << "Incorrect matrix type, matrix is symmetric ...\n";
      status = false;
    }

    if (!A->expanded()) {
      std::cout << "Incorrect matrix type, matrix is expanded ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, symmetric_expanded_coo_matrix_rows_, symmetric_expanded_coo_matrix_cols_, symmetric_expanded_coo_matrix_vals_);

    delete A;
    A = nullptr;

    return status.report(__func__);
  }


  TestOutcome csrMatrixImport()
  {
    TestStatus status;
    status = true;

    bool is_expand_symmetric = true;
    std::istringstream file(symmetric_duplicates_coo_matrix_file_);
    ReSolve::matrix::Csr* B = ReSolve::io::createCsrFromFile(file, is_expand_symmetric);

    index_type nnz_answer = static_cast<index_type>(symmetric_expanded_csr_matrix_vals_.size());
    if (B->getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if (!B->symmetric()) {
      std::cout << "Incorrect matrix type, matrix is symmetric ...\n";
      status = false;
    }

    if (!B->expanded()) {
      std::cout << "Incorrect matrix type, matrix is expanded ...\n";
      status = false;
    }

    status *= verifyAnswer(*B, symmetric_expanded_csr_matrix_rows_, symmetric_expanded_csr_matrix_cols_, symmetric_expanded_csr_matrix_vals_);

    delete B;
    B = nullptr;

    return status.report(__func__);
  }


  TestOutcome cooMatrixExport()
  {
    TestStatus status;
    status = true;

    // Read string into istream and status it to `createCooFromFile` function.
    std::ostringstream buffer;

    // Deep copy constant test vectors with matrix data to nonconstant ones
    std::vector<index_type> rows = general_coo_matrix_rows_;
    std::vector<index_type> cols = general_coo_matrix_cols_;
    std::vector<real_type>  vals = general_coo_matrix_vals_;

    // Get number of matrix rows
    const index_type N = 1 + *(std::max_element(rows.begin(), rows.end()));

    // Get number of matrix columns
    const index_type M = 1 + *(std::max_element(cols.begin(), cols.end()));

    // Get number of nonzeros
    const index_type NNZ = static_cast<index_type>(general_coo_matrix_vals_.size());

    // Create the test COO matrix
    ReSolve::matrix::Coo A(N, M, NNZ, false, false);
    A.setDataPointers(&rows[0],
                      &cols[0],
                      &vals[0],
                      memory::HOST);

    // Write the matrix to an ostream
    ReSolve::io::writeMatrixToFile(&A, buffer);
    status *= (buffer.str() == resolve_general_coo_matrix_file_);

    return status.report(__func__);
  }

  TestOutcome csrMatrixExport()
  {
    TestStatus status;
    status = true;

    // Read string into istream and status it to `createCooFromFile` function.
    std::ostringstream buffer;

    // Deep copy constant test vectors with matrix data to nonconstant ones
    std::vector<index_type> rows = general_csr_matrix_rows_;
    std::vector<index_type> cols = general_csr_matrix_cols_;
    std::vector<real_type>  vals = general_csr_matrix_vals_;

    // Get number of matrix rows
    const index_type N = static_cast<index_type>(rows.size()) - 1;

    // Get number of matrix columns
    const index_type M = 1 + *(std::max_element(cols.begin(), cols.end()));

    // Get number of nonzeros
    const index_type NNZ = static_cast<index_type>(vals.size());

    // Create the test CSR matrix
    ReSolve::matrix::Csr A(N, M, NNZ, false, false);
    A.setDataPointers(&rows[0],
                    &cols[0],
                    &vals[0],
                    memory::HOST);

    // Write the matrix to an ostream
    ReSolve::io::writeMatrixToFile(&A, buffer);
    status *= (buffer.str() == resolve_row_sorted_general_coo_matrix_file_);

    return status.report(__func__);
  }

  TestOutcome cooMatrixReadAndUpdate()
  {
    TestStatus status;

    bool is_symmetric = true;
    bool is_expanded  = false;

    // Create a 5x5 COO matrix with 10 nonzeros
    ReSolve::matrix::Coo A(5, 5, 10, is_symmetric, is_expanded);
    A.allocateMatrixData(memory::HOST);

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file2(symmetric_coo_matrix_file_);

    // Update matrix A with data from the matrix market file
    ReSolve::io::updateMatrixFromFile(file2, &A);

    // Check if the matrix data was correctly loaded
    status = true;

    index_type nnz_answer = static_cast<index_type>(symmetric_coo_matrix_vals_.size());
    if (A.getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(A, symmetric_coo_matrix_rows_, symmetric_coo_matrix_cols_, symmetric_coo_matrix_vals_);

    // Read next matrix market file into istream. This matrix has duplicates
    // that need to be merged and number of nonzeros needs to be recalculated
    // accordingly.
    std::istringstream file(symmetric_duplicates_coo_matrix_file_);

    // Update matrix A with data from the matrix market file
    ReSolve::io::updateMatrixFromFile(file, &A);

    if (A.getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(A, symmetric_coo_matrix_rows_, symmetric_coo_matrix_cols_, symmetric_coo_matrix_vals_);

    // Create a 5x5 COO matrix with 13 nonzeros
    is_expanded = true;
    ReSolve::matrix::Coo B(5, 5, 13, is_symmetric, is_expanded);
    B.allocateMatrixData(memory::HOST);

    // Read in symmetric matrix data
    std::istringstream file3(symmetric_duplicates_coo_matrix_file_);

    // Update matrix B with data from the matrix market file
    ReSolve::io::updateMatrixFromFile(file3, &B);

    nnz_answer = static_cast<index_type>(symmetric_expanded_coo_matrix_vals_.size());
    if (B.getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(B, symmetric_expanded_coo_matrix_rows_, symmetric_expanded_coo_matrix_cols_, symmetric_expanded_coo_matrix_vals_);

    return status.report(__func__);
  }


  TestOutcome csrMatrixReadAndUpdate()
  {
    TestStatus status;
    status = true;

    bool is_symmetric = true;
    bool is_expanded  = true;

    ReSolve::matrix::Csr A(5, 5, 13, is_symmetric, is_expanded);
    A.allocateMatrixData(memory::HOST);

    // Read in symmetric matrix data
    std::istringstream file(symmetric_duplicates_coo_matrix_file_);

    // Update matrix B with data from the matrix market file
    ReSolve::io::updateMatrixFromFile(file, &A);

    index_type nnz_answer = static_cast<index_type>(symmetric_expanded_csr_matrix_vals_.size());
    if (A.getNnz() != nnz_answer) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      std::cout << A.getNnz() << " ?= " << nnz_answer << "\n";
      status = false;
    }

    status *= verifyAnswer(A,
                           symmetric_expanded_csr_matrix_rows_,
                           symmetric_expanded_csr_matrix_cols_,
                           symmetric_expanded_csr_matrix_vals_);

    return status.report(__func__);
  }

  TestOutcome rhsVectorReadFromFile()
  {
    TestStatus status;

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file(general_vector_file_);

    // Create rhs vector and load its data from the input file
    vector::Vector* rhs = ReSolve::io::createVectorFromFile(file);

    // Check if the matrix data was correctly loaded
    status = true;

    const real_type* rhs_data = rhs->getData(memory::HOST);
    for (size_t i = 0; i < general_vector_vals_.size(); ++i) {
      if (!isEqual(rhs_data[i], general_vector_vals_[i]))
      {
        std::cout << "Incorrect vector value at storage element " << i << ".\n";
        status = false;
        break;
      }
      // std::cout << i << ": " << rhs[i] << "\n";
    }

    // Delete test vector
    delete rhs;

    return status.report(__func__);
  }

  TestOutcome rhsArrayReadFromFile()
  {
    TestStatus status;

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file(general_vector_file_);

    // Create rhs vector and load its data from the input file
    real_type* rhs = ReSolve::io::createArrayFromFile(file);

    // Check if the matrix data was correctly loaded
    status = true;

    for (size_t i = 0; i < general_vector_vals_.size(); ++i) {
      if (!isEqual(rhs[i], general_vector_vals_[i]))
      {
        std::cout << "Incorrect vector value at storage element " << i << ".\n";
        status = false;
        break;
      }
      // std::cout << i << ": " << rhs[i] << "\n";
    }

    return status.report(__func__);
  }

  TestOutcome rhsArrayReadAndUpdate()
  {
    TestStatus status;

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file(general_vector_file_);

    // For now let's test only the case when `updateArrayFromFile` does not allocate rhs
    real_type* rhs = new real_type[5]; //nullptr;

    // Update matrix A with data from the matrix market file
    ReSolve::io::updateArrayFromFile(file, &rhs);

    // Check if the matrix data was correctly loaded
    status = true;

    for (size_t i = 0; i < general_vector_vals_.size(); ++i) {
      if (!isEqual(rhs[i], general_vector_vals_[i]))
      {
        std::cout << "Incorrect vector value at storage element " << i << ".\n";
        status = false;
        break;
      }
      // std::cout << i << ": " << rhs[i] << "\n";
    }

    return status.report(__func__);
  }

  TestOutcome rhsVectorReadAndUpdate()
  {
    TestStatus status;

    // Read string into istream and status it to `createCooFromFile` function.
    std::istringstream file(general_vector_file_);

    // For now let's test only the case when `updateArrayFromFile` does not allocate rhs
    index_type N = static_cast<index_type>(general_vector_vals_.size());
    vector::Vector vec_rhs(N);
    vec_rhs.allocate(memory::HOST);

    // Update vector vec_rhs with data from the matrix market file
    ReSolve::io::updateVectorFromFile(file, &vec_rhs);

    // Check if the vector data was correctly loaded
    status = true;

    real_type* rhs = vec_rhs.getData(memory::HOST);
    for (size_t i = 0; i < general_vector_vals_.size(); ++i) {
      if (!isEqual(rhs[i], general_vector_vals_[i]))
      {
        std::cout << "Incorrect vector value at storage element " << i << ".\n";
        status = false;
        break;
      }
      // std::cout << i << ": " << rhs[i] << "\n";
    }

    return status.report(__func__);
  }

private:
  bool verifyAnswer(/* const */ ReSolve::matrix::Coo& answer,
                    const std::vector<index_type>& row_data,
                    const std::vector<index_type>& col_data,
                    const std::vector<real_type>& val_data)
  {
    for (size_t i = 0; i < val_data.size(); ++i) {
      if ((answer.getRowData(memory::HOST)[i] != row_data[i]) ||
          (answer.getColData(memory::HOST)[i] != col_data[i]) ||
          (!isEqual(answer.getValues(memory::HOST)[i], val_data[i])))
      {
        std::cout << "Incorrect matrix value at storage element " << i << ".\n";
        return false;
      }
    }
    return true;
  }

    bool verifyAnswer(/* const */ ReSolve::matrix::Csr& answer,
                      const std::vector<index_type>& row_data,
                      const std::vector<index_type>& col_data,
                      const std::vector<real_type>& val_data)
    {
      for (size_t i = 0; i < val_data.size(); ++i) {
        if ((answer.getColData(memory::HOST)[i] != col_data[i]) ||
            (!isEqual(answer.getValues(memory::HOST)[i], val_data[i]))) {
          std::cout << "Incorrect matrix value at storage element " << i << ".\n";
          return false;
        }          
      }
  
      for (size_t i = 0; i < row_data.size(); ++i) {
        if(answer.getRowData(memory::HOST)[i] != row_data[i]) {
          std::cout << "Incorrect row pointer value at storage element " << i << ".\n";
          return false;
        }
      }
  
      return true;
    }

private:
  //
  // Test examples
  //

  /// String pretending to be matrix market file
  /// Same stored in file `matrix_general_coo_ordered.mtx`
  const std::string general_coo_matrix_file_ = 
R"(%%MatrixMarket matrix coordinate real general
% This ASCII file represents a sparse MxN matrix with L 
% nonzeros in the following Matrix Market format:
%
% +----------------------------------------------+
% |%%MatrixMarket matrix coordinate real general | <--- header line
% |%                                             | <--+
% |% comments                                    |    |-- 0 or more comment lines
% |%                                             | <--+         
% |    M  N  L                                   | <--- rows, columns, entries
% |    I1  J1  A(I1, J1)                         | <--+
% |    I2  J2  A(I2, J2)                         |    |
% |    I3  J3  A(I3, J3)                         |    |-- L lines
% |        . . .                                 |    |
% |    IL JL  A(IL, JL)                          | <--+
% +----------------------------------------------+   
%
% Indices are 1-based, i.e. A(1,1) is the first element.
%
%=================================================================================
  5  5  8
    1     1   1.000e+00
    2     2   1.050e+01
    3     3   1.500e-02
    1     4   6.000e+00
    4     2   2.505e+02
    4     4  -2.800e+02
    4     5   3.332e+01
    5     5   1.200e+01
)";

  /// String pretending to be matrix market file.
  /// Generated by ReSolve's `writeMatrixToFile` function.
  const std::string resolve_general_coo_matrix_file_ = 
R"(%%MatrixMarket matrix coordinate real general
% Generated by Re::Solve <https://github.com/ORNL/ReSolve>
5 5 8
1 1 1.000000000000000e+00
1 4 6.000000000000000e+00
2 2 1.050000000000000e+01
3 3 1.500000000000000e-02
4 2 2.505000000000000e+02
4 4 -2.800000000000000e+02
4 5 3.332000000000000e+01
5 5 1.200000000000000e+01
)";

  /// String pretending to be matrix market file.
  /// Generated by ReSolve's `writeMatrixToFile` function.
  const std::string resolve_row_sorted_general_coo_matrix_file_ = 
R"(%%MatrixMarket matrix coordinate real general
% Generated by Re::Solve <https://github.com/ORNL/ReSolve>
5 5 8
1 1 1.000000000000000e+00
1 4 6.000000000000000e+00
2 2 1.050000000000000e+01
3 3 1.500000000000000e-02
4 2 2.505000000000000e+02
4 4 -2.800000000000000e+02
4 5 3.332000000000000e+01
5 5 1.200000000000000e+01
)";

  /// Matching COO matrix data as it is supposed to be read from the file
  const std::vector<index_type> general_coo_matrix_rows_ = {0,0,1,2,3,3,3,4};
  const std::vector<index_type> general_coo_matrix_cols_ = {0,3,1,2,1,3,4,4};
  const std::vector<real_type> general_coo_matrix_vals_  = { 1.000e+00,
                                                             6.000e+00,
                                                             1.050e+01,
                                                             1.500e-02,
                                                             2.505e+02,
                                                            -2.800e+02,
                                                             3.332e+01,
                                                             1.200e+01 };

  /// Matching CSR matrix data as it is supposed to be read from the file
  const std::vector<index_type> general_csr_matrix_rows_ = {0,2,3,4,7,8};
  const std::vector<index_type> general_csr_matrix_cols_ = {0,3,1,2,1,3,4,4};
  const std::vector<real_type>  general_csr_matrix_vals_ = { 1.000e+00,
                                                             6.000e+00,
                                                             1.050e+01,
                                                             1.500e-02,
                                                             2.505e+02,
                                                            -2.800e+02,
                                                             3.332e+01,
                                                             1.200e+01 };

  //
  //     [11          15]
  //     [   22 23 24   ]
  // A = [      33    35]
  //     [         44   ]
  //     [            55]
  //
  const std::string symmetric_coo_matrix_file_ =
R"(%%MatrixMarket matrix coordinate real symmetric
% This ASCII file represents a sparse MxN matrix with L 
% nonzeros in the following Matrix Market format:
%
% +------------------------------------------------+
% |%%MatrixMarket matrix coordinate real symmetric | <--- header line
% |%                                               | <--+
% |% comments                                      |    |-- 0 or more comment lines
% |%                                               | <--+         
% |    M  N  L                                     | <--- rows, columns, entries
% |    I1  J1  A(I1, J1)                           | <--+
% |    I2  J2  A(I2, J2)                           |    |
% |    I3  J3  A(I3, J3)                           |    |-- L lines
% |        . . .                                   |    |
% |    IL JL  A(IL, JL)                            | <--+
% +------------------------------------------------+   
%
% Indices are 1-based, i.e. A(1,1) is the first element.
%
%=================================================================================
%
 5  5  9
 1  1  11.0
 1  5  15.0
 2  2  22.0 
 2  3  23.0  
 2  4  24.0 
 3  3  33.0 
 3  5  35.0
 4  4  44.0   
 5  5  55.0
 )";


  //
  //     [11          15]
  //     [   22 23 24   ]
  // A = [      33    35]
  //     [         44   ]
  //     [            55]
  //
  // A(1,1), A(5,5) and A(2,4) are stored in duplicate entries.
  const std::string symmetric_duplicates_coo_matrix_file_ =
R"(%%MatrixMarket matrix coordinate real symmetric
%
 5  5  13
 5  5  50.0
 1  1  10.0
 1  5  15.0
 2  2  22.0 
 2  3  23.0  
 2  4  20.0 
 3  3  33.0 
 3  5  35.0
 4  4  44.0   
 5  5   5.0
 2  4   2.0 
 2  4   2.0 
 1  1   1.0
 )";


  /// Matching COO matrix data as it is supposed to be read from the file
  const std::vector<index_type> symmetric_coo_matrix_rows_ = {0,0,1,1,1,2,2,3,4};
  const std::vector<index_type> symmetric_coo_matrix_cols_ = {0,4,1,2,3,2,4,3,4};
  const std::vector<real_type>  symmetric_coo_matrix_vals_ = {11.0,
                                                              15.0,
                                                              22.0,
                                                              23.0,
                                                              24.0,
                                                              33.0,
                                                              35.0,
                                                              44.0,
                                                              55.0};

    // Matching expanded COO matrix data as it is supposed to be created
    //
    //     [11          15]
    //     [   22 23 24   ]
    // A = [   23 33    35]
    //     [   24    44   ]
    //     [15    35    55]
    //
    // Symmetric matrix stored in COO general format
    //
    const std::vector<index_type> symmetric_expanded_coo_matrix_rows_ = {0,0,1,1,1,2,2,2,3,3,4,4,4};
    const std::vector<index_type> symmetric_expanded_coo_matrix_cols_ = {0,4,1,2,3,1,2,4,1,3,0,2,4};
    const std::vector<real_type>  symmetric_expanded_coo_matrix_vals_ = { 11.0,
                                                                          15.0,
                                                                          22.0,
                                                                          23.0,
                                                                          24.0,
                                                                          23.0,
                                                                          33.0,
                                                                          35.0,
                                                                          24.0,
                                                                          44.0,
                                                                          15.0,
                                                                          35.0,
                                                                          55.0 };

    // Matching expanded CSR matrix data as it is supposed to be converted
    //
    //     [11          15]
    //     [   22 23 24   ]
    // A = [   23 33    35]
    //     [   24    44   ]
    //     [15    35    55]
    //
    // Symmetric matrix in CSR general format
    //
    const std::vector<index_type> symmetric_expanded_csr_matrix_rows_ = {0,2,5,8,10,13};
    const std::vector<index_type> symmetric_expanded_csr_matrix_cols_ = {0,4,1,2,3,1,2,4,1,3,0,2,4};
    const std::vector<real_type>  symmetric_expanded_csr_matrix_vals_ = { 11.0,
                                                                          15.0,
                                                                          22.0,
                                                                          23.0,
                                                                          24.0,
                                                                          23.0,
                                                                          33.0,
                                                                          35.0,
                                                                          24.0,
                                                                          44.0,
                                                                          15.0,
                                                                          35.0,
                                                                          55.0 };

  const std::string general_vector_file_ = 
R"(% This ASCII file represents a sparse MxN matrix with L 
% nonzeros in the following Matrix Market format:
%
%
%=================================================================================
  5  1
   1.000e+00
   2.000e+01
   3.000e-02
   4.000e+00
   5.505e+02
)";

  const std::vector<real_type> general_vector_vals_ = { 1.000e+00,
                                                        2.000e+01,
                                                        3.000e-02,
                                                        4.000e+00,
                                                        5.505e+02 };

  /// Location of other test data
  std::string datafiles_folder_;
}; // class MatrixIoTests

}} // namespace ReSolve::tests
