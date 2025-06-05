#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <cassert>
#include <tests/unit/TestBase.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve { namespace tests {
  class SparseTests : TestBase
  {
  public:
    SparseTests(memory::MemorySpace memspace = memory::HOST) : memspace_(memspace)
    {

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
     * @param[in] row_density - number of non-zeros per row
     * @param[in] is_symmetric - true if the matrix is symmetric, false otherwise
     * @param[in] is_expanded - true if the matrix is expanded (all non-zeros
     are stored explicitly, not assumed based on symmetry), false otherwise
     *
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome constructor(index_type n,
                            index_type m,
                            index_type nnz,
                            bool is_symmetric = false,
                            bool is_expanded = true)
    {
      TestStatus success;
      success = true;

      ReSolve::matrix::Csr A(n, m, nnz, is_symmetric, is_expanded);

      if (A.getNumRows() != n || A.getNumColumns() != m || A.getNnz() != nnz) {
        std::cout << "Matrix dimensions do not match expected values.\n";
        success = false;
      }

      return success.report(__func__);
    }

    /**
     * @brief Test set data pointers for the sparse matrix
     *
     * Sets the row, column, and value data pointers for the sparse matrix and
     * checks if the pointers are set correctly.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     *
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome setDataPointers(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr A(n, m, nnz);

      index_type* h_row_data = new index_type[n + 1];
      for (index_type i = 0; i <= n; ++i) {
        h_row_data[i] = i * (nnz / n); // Simple pattern for row pointers
      }
      index_type* h_col_data = new index_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        h_col_data[i] = i % m; // Simple pattern for column indices
      }
      real_type* h_val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        h_val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A.setDataPointers(h_row_data, h_col_data, h_val_data, memory::HOST) != 0) {
        std::cout << "Failed to set data pointers.\n";
        success = false;
      } else if (A.getRowData(memory::HOST) != h_row_data ||
                A.getColData(memory::HOST) != h_col_data ||
                A.getValues(memory::HOST) != h_val_data) {
        std::cout << "Data pointers do not point to expected values.\n";
        success = false;
      }

      return success.report(__func__);
    }

    /**
     * @brief Test setting values pointer for the sparse matrix
     *
     * Sets the values pointer for the sparse matrix and checks if the pointer
     * points to the expected values.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     *
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome setValuesPointer(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr A(n, m, nnz);

      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A.setValuesPointer(val_data, memory::HOST) != 0) {
        std::cout << "Failed to set values pointer.\n";
        success = false;
      } else if (A.getValues(memory::HOST) != val_data) {
        std::cout << "Values pointer does not point to expected values.\n";
        success = false;
      }

      return success.report(__func__);
    }

    /**
     * @brief Test copying values into the sparse matrix
     *
     * Copies values into the sparse matrix, modifies the original values,
     * and verifies the correct unmodified values are stored in the matrix,
     * and that the values pointer does not point to the original array.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     *
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome copyValues(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, m, nnz);

      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }
      real_type* h_val_data = nullptr;
      if (A->copyValues(val_data, memory::HOST, memspace_) != 0) {
        std::cout << "Failed to copy values.\n";
        success = false;
      } else {
        // Modify original values to ensure copy worked
        for (index_type i = 0; i < nnz; ++i) {
          val_data[i] *= 2; // Change the values
        }

        if (A->getValues(memspace_) == nullptr) {
          std::cout << "Values pointer is null after copy.\n";
          success = false;
        } else {
          if (memspace_ == memory::HOST) {
            h_val_data = A->getValues(memory::HOST);
          } else {
            real_type* d_val_data = A->getValues(memory::DEVICE);
            h_val_data = new real_type[nnz];
            mem_.copyArrayDeviceToHost(h_val_data, d_val_data, nnz);
            if (h_val_data == nullptr) {
              std::cout << "Failed to copy values from device to host.\n";
              success = false;
            }
          }

          // Check if the copied values are correct
          for (index_type i = 0; i < nnz; ++i) {
            if (h_val_data[i] != static_cast<real_type>(i + 1)) {
              std::cout << "Copied values do not match expected values.\n";
              success = false;
              break;
            }
          }
        }
      }

      // Clean up allocated memory
      delete[] val_data;

      if (A->destroyMatrixData(memspace_) != 0) {
        std::cout << "Failed to destroy matrix data.\n";
        success = false;
      }

      return success.report(__func__);
    }

    /**
     * @brief Verify that matrix will not reset data pointers when it currently owns data.
     *
     * Copies values into the sparse matrix and then attempts to set the values
     * pointer, which should fail since the matrix owns the data.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome copyValuesAndSetDataPointers(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, m, nnz);

      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A->copyValues(val_data, memory::HOST, memspace_) != 0) {
        std::cout << "Failed to copy values.\n";
        success = false;
      }

      index_type* row_data = new index_type[n + 1];
      for (index_type i = 0; i <= n; ++i) {
        row_data[i] = i * (nnz / n); // Simple pattern for row pointers
      }
      index_type* col_data = new index_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        col_data[i] = i % m; // Simple pattern for column indices
      }

      if (A->setDataPointers(row_data, col_data, val_data, memspace_) == 0) {
        std::cout << "Should not have set data pointers after copying values.\n";
        success = false;
      }

      // Clean up allocated memory
      delete[] val_data;
      delete[] row_data;
      delete[] col_data;

      return success.report(__func__);
    }

    /**
     * @brief Verify that matrix will not reset values pointer when it currently owns data.
     *
     * Copies a data array using `copyValues`, then attempts to set the values pointer,
     * and verifies that this results in an error.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome copyValuesAndSetValues(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, m, nnz);

      real_type* val_data = new real_type[nnz];
      for (index_type i = 0; i < nnz; ++i) {
        val_data[i] = static_cast<real_type>(i + 1);
      }

      if (A->copyValues(val_data, memory::HOST, memspace_) != 0) {
        std::cout << "Failed to copy values.\n";
        success = false;
      }

      if (A->setValuesPointer(val_data, memspace_) == 0) {
        std::cout << "Should not have set values pointer when matrix owns data.\n";
        success = false;
      }

      // Clean up allocated memory
      delete[] val_data;

      return success.report(__func__);
    }

    /**
     * @brief Test allocating and destroying matrix data
     *
     * Allocates memory for the sparse matrix data and checks if the pointers
     * are not null after allocation. Then destroys the allocated data.
     *
     * @param[in] n - number of rows
     * @param[in] m - number of columns
     * @param[in] row_density - number of non-zeros per row
     *
     * @return TestOutcome indicating success or failure of the test
     */
    TestOutcome allocateAndDestroyData(index_type n, index_type m, index_type row_density)
    {
      TestStatus success;
      success = true;

      index_type nnz = n * row_density; // Total non-zeros based on row density

      ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, m, nnz);

      if (A->allocateMatrixData(memspace_) != 0) {
        std::cout << "Failed to allocate matrix data.\n";
        success = false;
      } else if (A->getRowData(memspace_) == nullptr ||
                 A->getColData(memspace_) == nullptr ||
                 A->getValues(memspace_) == nullptr) {
        std::cout << "Matrix data pointers are null after allocation.\n";
        success = false;
      }

      if (A->destroyMatrixData(memspace_) != 0) {
        std::cout << "Failed to destroy matrix data.\n";
        success = false;
      } else if (A->getRowData(memspace_) != nullptr ||
                 A->getColData(memspace_) != nullptr ||
                 A->getValues(memspace_) != nullptr) {
        std::cout << "Matrix data pointers are not null after destruction.\n";
        success = false;
      }

      return success.report(__func__);
    }

    private:
      memory::MemorySpace memspace_;
      MemoryHandler mem_;
  }; // class SparseTests
}} // namespace ReSolve::tests