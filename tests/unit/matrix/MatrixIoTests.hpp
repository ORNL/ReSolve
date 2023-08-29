#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { namespace tests {

class MatrixIoTests : TestBase
{
public:
  MatrixIoTests(){}
  virtual ~MatrixIoTests(){}


  TestOutcome cooMatrixImport()
  {
    TestStatus status;

    // Read string into istream and status it to `readMatrixFromFile` function.
    std::istringstream file(general_coo_matrix_file_);
    ReSolve::matrix::Coo* A = ReSolve::io::readMatrixFromFile(file);

    // Check if the matrix data was correctly loaded
    status = true;

    if(A->getNnz() != general_coo_matrix_vals_.size()) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if(A->symmetric()) {
      std::cout << "Incorrect matrix type, matrix not symmetric ...\n";
      status = false;
    }

    if(!A->expanded()) {
      std::cout << "Incorrect matrix type, matrix is general (expanded) ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, general_coo_matrix_rows_, general_coo_matrix_cols_, general_coo_matrix_vals_);

    // A->print();
    delete A;
    A = nullptr;

    std::istringstream file2(symmetric_coo_matrix_file_);
    A = ReSolve::io::readMatrixFromFile(file2);

    if(A->getNnz() != symmetric_coo_matrix_vals_.size()) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    if(!A->symmetric()) {
      std::cout << "Incorrect matrix type, matrix is symmetric ...\n";
      status = false;
    }

    if(A->expanded()) {
      std::cout << "Incorrect matrix type, matrix not expanded ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, symmetric_coo_matrix_rows_, symmetric_coo_matrix_cols_, symmetric_coo_matrix_vals_);

    return status.report(__func__);
  }

  TestOutcome cooMatrixImport2(std::ifstream& file)
  {
    TestStatus status;
    ReSolve::matrix::Coo* A = ReSolve::io::readMatrixFromFile(file);
    if (A->getNnz() != general_coo_matrix_vals_.size()) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(*A, general_coo_matrix_rows_, general_coo_matrix_cols_, general_coo_matrix_vals_);

    return status.report(__func__);
  }

  TestOutcome cooMatrixReadAndUpdate()
  {
    TestStatus status;

    // Create a 5x5 COO matrix with 10 nonzeros
    ReSolve::matrix::Coo A(5, 5, 10);
    A.allocateMatrixData("cpu");

    // Read string into istream and status it to `readMatrixFromFile` function.
    std::istringstream file2(symmetric_coo_matrix_file_);

    // Update matrix A with data from the matrix market file
    ReSolve::io::readAndUpdateMatrix(file2, &A);

    // Check if the matrix data was correctly loaded
    status = true;

    if(A.getNnz() != symmetric_coo_matrix_vals_.size()) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(A, symmetric_coo_matrix_rows_, symmetric_coo_matrix_cols_, symmetric_coo_matrix_vals_);

    // Read string into istream and status it to `readMatrixFromFile` function.
    std::istringstream file(general_coo_matrix_file_);

    // Update matrix A with data from the matrix market file
    ReSolve::io::readAndUpdateMatrix(file, &A);

    if(A.getNnz() != general_coo_matrix_vals_.size()) {
      std::cout << "Incorrect NNZ read from the file ...\n";
      status = false;
    }

    status *= verifyAnswer(A, general_coo_matrix_rows_, general_coo_matrix_cols_, general_coo_matrix_vals_);

    return status.report(__func__);
  }

  TestOutcome rhsVectorReadFromFile()
  {
    TestStatus status;

    // Read string into istream and status it to `readMatrixFromFile` function.
    std::istringstream file(general_vector_file_);

    // Create rhs vector and load its data from the input file
    real_type* rhs = ReSolve::io::readRhsFromFile(file);

    // Check if the matrix data was correctly loaded
    status = true;

    for(index_type i = 0; i < general_vector_vals_.size(); ++i) {
      if(!isEqual(rhs[i], general_vector_vals_[i]))
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

    // Read string into istream and status it to `readMatrixFromFile` function.
    std::istringstream file(general_vector_file_);

    // For now let's test only the case when `readAndUpdateRhs` does not allocate rhs
    real_type* rhs = new real_type[5]; //nullptr;

    // Update matrix A with data from the matrix market file
    ReSolve::io::readAndUpdateRhs(file, &rhs);

    // Check if the matrix data was correctly loaded
    status = true;

    for(index_type i = 0; i < general_vector_vals_.size(); ++i) {
      if(!isEqual(rhs[i], general_vector_vals_[i]))
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
    for(index_type i = 0; i < val_data.size(); ++i) {
      if((answer.getRowData("cpu")[i] != row_data[i]) ||
         (answer.getColData("cpu")[i] != col_data[i]) ||
         (!isEqual(answer.getValues("cpu")[i], val_data[i])))
      {
        std::cout << "Incorrect matrix value at storage element " << i << ".\n";
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
R"(% This ASCII file represents a sparse MxN matrix with L 
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

  /// Matching COO matrix data as it is supposed to be read from the file
  const std::vector<index_type> general_coo_matrix_rows_ = {0,1,2,0,3,3,3,4};
  const std::vector<index_type> general_coo_matrix_cols_ = {0,1,2,3,1,3,4,4};
  const std::vector<real_type> general_coo_matrix_vals_ = { 1.000e+00,
                                                            1.050e+01,
                                                            1.500e-02,
                                                            6.000e+00,
                                                            2.505e+02,
                                                           -2.800e+02,
                                                            3.332e+01,
                                                            1.200e+01 };

  const std::string symmetric_coo_matrix_file_ =
R"(%%MatrixMarket matrix coordinate real symmetric
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


  /// Matching COO matrix data as it is supposed to be read from the file
  const std::vector<index_type> symmetric_coo_matrix_rows_ = {0,0,1,1,1,2,2,3,4};
  const std::vector<index_type> symmetric_coo_matrix_cols_ = {0,4,1,2,3,2,4,3,4};
  const std::vector<real_type> symmetric_coo_matrix_vals_ = { 11.0,
                                                              15.0,
                                                              22.0,
                                                              23.0,
                                                              24.0,
                                                              33.0,
                                                              35.0,
                                                              44.0,
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
