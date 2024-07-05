#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <tests/unit/TestBase.hpp>
#include <resolve/matrix/Utilities.hpp>

namespace ReSolve { namespace tests {

class MatrixConversionTests : TestBase
{
  public:
    MatrixConversionTests(){}
    virtual ~MatrixConversionTests(){}
  
    TestOutcome newCooToCsr()
    {
      TestStatus status;
      status = true;
  
      matrix::Coo* A = createSymmetricCooMatrix(); 
      ReSolve::matrix::Csr* A_csr = new matrix::Csr(A->getNumRows(), A->getNumColumns(), 0);
  
      int retval = coo2csr_new(A, A_csr, memory::HOST);
  
      status *= verifyAnswer(*A_csr, symmetric_expanded_csr_matrix_rows_, symmetric_expanded_csr_matrix_cols_, symmetric_expanded_csr_matrix_vals_);
  
      delete A;
      delete A_csr;
  
      return status.report(__func__);
    }
  
    TestOutcome oldCooToCsr()
    {
      TestStatus status;
      status.expectFailure();
  
      matrix::Coo* A = createSymmetricCooMatrix(); 
      ReSolve::matrix::Csr* A_csr = new matrix::Csr(A->getNumRows(), A->getNumColumns(), 0);
  
      int retval = coo2csr(A, A_csr, memory::HOST);
  
      status *= verifyAnswer(*A_csr, symmetric_expanded_csr_matrix_rows_, symmetric_expanded_csr_matrix_cols_, symmetric_expanded_csr_matrix_vals_);
  
      delete A;
      delete A_csr;
  
      return status.report(__func__);
    }
  
  private:
  
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
  
    //
    // Test examples
    //
  
  
    //
    //     [11          15]
    //     [   22 23 24   ]
    // A = [      33    35]
    //     [         44   ]
    //     [            55]
    //
    // Symmetric matrix in COO unordered format
    // Only upper triangular matrix is stored
    // A(2,4) is stored in two duplicate entries
    //
    matrix::Coo* createSymmetricCooMatrix()
    {
      matrix::Coo* A = new matrix::Coo(5, 5, 10, true, false);
      index_type rows[10] = {0, 0, 1, 1, 1, 2, 2, 3, 4, 1};
      index_type cols[10] = {0, 4, 3, 1, 2, 4, 2, 3, 4, 3};
      real_type  vals[10] = {11.0, 15.0,12.0, 22.0, 23.0, 35.0, 33.0, 44.0, 55.0, 12.0};
  
      A->allocateMatrixData(memory::HOST);
      A->updateData(rows, cols, vals, memory::HOST, memory::HOST);
      return A;
    }
  
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
}; // class MatrixConversionTests

}} // namespace ReSolve::tests
