#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <tests/unit/TestBase.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <cassert>

namespace ReSolve { namespace tests {
  class SparseTests : TestBase
  {
  public:
    SparseTests(ReSolve::MatrixHandler& handler) : handler_(handler)
    {
      if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled()) {
        memspace_ = memory::DEVICE;
      } else {
        memspace_ = memory::HOST;
      }
    }

    ~SparseTests()
    {
    }

    /**
     * @brief Constructor test for sparse matrix
     * 
     * Constructs a CSR matrix with the given parameters and confirms that the
     * parameters are stored correctly.
     * 
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] nnz - number of non-zeros
     * @param[in] is_symmetric - true if the matrix is symmetric, false otherwise
     * @param[in] is_expanded - true if the matrix is expanded, false otherwise
     * 
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome constructor(index_type n, 
                            index_type m, 
                            index_type nnz,
                            bool is_symmetric = false,
                            bool is_expanded = true) 
    {
      TestStatus status;
      status = true;

      ReSolve::matrix::Csr A(n, m, nnz, is_symmetric, is_expanded);

      if (A.getNumRows() != n || A.getNumColumns() != m || A.getNnz() != nnz) {
        std::cout << "Matrix dimensions do not match expected values.\n";
        status = false;
      }

      return status.report(__func__);
    }

    /**
     * @brief Test set data pointers for the sparse matrix
     * 
     * Sets the row, column, and value data pointers for the sparse matrix and
     * checks if the pointers are set correctly.
     * 
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome setDataPointers(index_type n, index_type m, index_type nnz)
    {
      assert(nnz % m == 0 && "For this test, nnz must be divisible by m");
      
      TestStatus status;
      status = true;

      ReSolve::matrix::Csr A(n, m, nnz);

      index_type* row_data = new index_type[n + 1];
      for (index_type i = 0; i <= n; ++i) {
        row_data[i] = i * (nnz / n); // Simple pattern for row pointers
      }
      index_type* col_data = new index_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        col_data[i] = i % m; // Simple pattern for column indices
      }
      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A.setDataPointers(row_data, col_data, val_data, memspace_) != 0) {
        std::cout << "Failed to set data pointers.\n";
        status = false;
      } else if (A.getRowData(memspace_) != row_data || 
                A.getColData(memspace_) != col_data || 
                A.getValues(memspace_) != val_data) {
        std::cout << "Data pointers do not point to expected values.\n";
        status = false;
      }

      return status.report(__func__);
    }

    /**
     * @brief Test setting values pointer for the sparse matrix
     * 
     * Sets the values pointer for the sparse matrix and checks if the pointer
     * points to the expected values.
     * 
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome setValuesPointer(index_type n, index_type m, index_type nnz)
    {
      TestStatus status;
      status = true;

      ReSolve::matrix::Csr A(n, m, nnz);

      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A.setValuesPointer(val_data, memspace_) != 0) {
        std::cout << "Failed to set values pointer.\n";
        status = false;
      } else if (A.getValues(memspace_) != val_data) {
        std::cout << "Values pointer does not point to expected values.\n";
        status = false;
      }

      return status.report(__func__);
    }

    /**
     * @brief Test allocating and destroying matrix data
     * 
     * Allocates memory for the sparse matrix data and checks if the pointers
     * are not null after allocation. Then destroys the allocated data.
     * 
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] nnz - number of non-zeros
     * 
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome allocateAndDestroyData(index_type n, index_type m, index_type nnz)
    {
      TestStatus status;
      status = true;

      ReSolve::matrix::Csr A(n, m, nnz);

      if (A.allocateMatrixData(memspace_) != 0) {
        std::cout << "Failed to allocate matrix data.\n";
        status = false;
      } else if (A.getRowData(memspace_) == nullptr || 
                 A.getColData(memspace_) == nullptr || 
                 A.getValues(memspace_) == nullptr) {
        std::cout << "Matrix data pointers are null after allocation.\n";
        status = false;
      }

      A.destroyMatrixData(memspace_);

      return status.report(__func__);
    }

    private:
      ReSolve::MatrixHandler& handler_;
      memory::MemorySpace memspace_;
  }; // class SparseTests
}} // namespace ReSolve::tests