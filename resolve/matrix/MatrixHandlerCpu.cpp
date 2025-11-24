#include "MatrixHandlerCpu.hpp"

#include <algorithm>
#include <cassert>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspaceCpu.hpp>
#include <resolve/workspace/ScaleAddBufferCpu.hpp>

namespace ReSolve
{
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  /**
   * @brief Empty constructor for MatrixHandlerCpu class.
   */
  MatrixHandlerCpu::MatrixHandlerCpu()
  {
  }

  /**
   * @brief Empty destructor for MatrixHandlerCpu class.
   */
  MatrixHandlerCpu::~MatrixHandlerCpu()
  {
  }

  /**
   * @brief Constructor for MatrixHandlerCpu class.
   * @param[in] new_workspace - pointer to LinAlgWorkspaceCpu object
   */
  MatrixHandlerCpu::MatrixHandlerCpu(LinAlgWorkspaceCpu* new_workspace)
  {
    workspace_ = new_workspace;
  }

  /**
   * @brief Marks when values have changed in MatrixHandlerCpu class.
   * @param[in] values_changed - boolean value indicating if values have changed
   */
  void MatrixHandlerCpu::setValuesChanged(bool values_changed)
  {
    values_changed_ = values_changed;
  }

  /**
   * @brief result := alpha * A * x + beta * result
   *
   * @param[in]     A - matrix
   * @param[in]     vec_x - vector multiplied by A
   * @param[in,out] vec_result - resulting vector
   * @param[in]     alpha - matrix-vector multiplication factor
   * @param[in]     beta - sum into result factor
   * @return int    error code, 0 if successful
   *
   * @pre Matrix `A` is in CSR format.
   *
   * @note If we decide to implement this function for different matrix
   * format, the check for CSR matrix will be replaced with a switch
   * statement to select implementation for recognized input matrix
   * format.
   */
  int MatrixHandlerCpu::matvec(matrix::Sparse*  A,
                               vector_type*     vec_x,
                               vector_type*     vec_result,
                               const real_type* alpha,
                               const real_type* beta)
  {
    using namespace constants;

    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    index_type* ia = A->getRowData(memory::HOST);
    index_type* ja = A->getColData(memory::HOST);
    real_type*  a  = A->getValues(memory::HOST);

    real_type* x_data      = vec_x->getData(memory::HOST);
    real_type* result_data = vec_result->getData(memory::HOST);
    real_type  sum;
    real_type  y;
    real_type  t;
    real_type  c;

    // Kahan algorithm for stability
    for (int i = 0; i < A->getNumRows(); ++i)
    {
      sum = 0.0;
      c   = 0.0;
      for (int j = ia[i]; j < ia[i + 1]; ++j)
      {
        y   = (a[j] * x_data[ja[j]]) - c;
        t   = sum + y;
        c   = (t - sum) - y;
        sum = t;
        //  sum += (a[j] * x_data[ja[j]]);
      }
      sum *= (*alpha);
      result_data[i] = result_data[i] * (*beta) + sum;
    }
    vec_result->setDataUpdated(memory::HOST);
    return 0;
  }

  /**
   * @brief Matrix infinity norm
   *
   * @param[in]  A - matrix
   * @param[out] norm - matrix norm
   * @return int error code, 0 if successful
   *
   * @pre Matrix `A` is in CSR format.
   *
   * @note If we decide to implement this function for different matrix
   * format, the check for CSR matrix will be replaced with a switch
   * statement to select implementation for recognized input matrix
   * format.
   */
  int MatrixHandlerCpu::matrixInfNorm(matrix::Sparse* A, real_type* norm)
  {
    using memory::HOST;
    assert(A->getSparseFormat() == matrix::Sparse::COMPRESSED_SPARSE_ROW && "Matrix has to be in CSR format for matrix-vector product.\n");

    real_type sum = 0.0;
    real_type nrm = 0.0;

    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      sum = 0.0;
      for (index_type j = A->getRowData(HOST)[i]; j < A->getRowData(HOST)[i + 1]; ++j)
      {
        sum += std::abs(A->getValues(HOST)[j]);
      }
      if (i == 0 || sum > nrm)
      {
        nrm = sum;
      }
    }
    *norm = nrm;
    return 0;
  }

  /**
   * @brief Convert CSC to CSR matrix on the host
   *
   * @authors Slaven Peles <peless@ornl.gov>, Daniel Reynolds (SMU), and
   * David Gardner and Carol Woodward (LLNL)
   *
   * @param[in]  A_csc - pointer to the CSC matrix
   * @param[out] A_csr - pointer to an allocated CSR matrix
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    assert(A_csc->getNnz() == A_csr->getNnz());
    assert(A_csc->getNumRows() == A_csr->getNumRows());
    assert(A_csc->getNumColumns() == A_csr->getNumColumns());
    index_type nnz = A_csc->getNnz();
    index_type n   = A_csc->getNumRows();
    index_type m   = A_csc->getNumColumns();

    index_type* rowIdxCsc = A_csc->getRowData(memory::HOST);
    index_type* colPtrCsc = A_csc->getColData(memory::HOST);
    real_type*  valuesCsc = A_csc->getValues(memory::HOST);

    index_type* rowPtrCsr = A_csr->getRowData(memory::HOST);
    index_type* colIdxCsr = A_csr->getColData(memory::HOST);
    real_type*  valuesCsr = A_csr->getValues(memory::HOST);

    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= n; ++i)
    {
      rowPtrCsr[i] = 0;
    }

    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i)
    {
      colIdxCsr[i] = 0;
      valuesCsr[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i)
    {
      rowPtrCsr[rowIdxCsc[i]]++;
    }

    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < n; ++row)
    {
      // Store value in row pointer to temp
      index_type temp = rowPtrCsr[row];

      // Copy cumulative sum to the row pointer
      rowPtrCsr[row] = rowsum;

      // Update row sum
      rowsum += temp;
    }
    rowPtrCsr[n] = nnz;

    for (index_type col = 0; col < m; ++col)
    {
      // Compute positions of column indices and values in CSR matrix and store them there
      // Overwrites CSR row pointers in the process
      for (index_type jj = colPtrCsc[col]; jj < colPtrCsc[col + 1]; jj++)
      {
        index_type row  = rowIdxCsc[jj];
        index_type dest = rowPtrCsr[row];

        colIdxCsr[dest] = col;
        valuesCsr[dest] = valuesCsc[jj];

        rowPtrCsr[row]++;
      }
    }

    // Restore CSR row pointer values
    for (index_type row = 0, last = 0; row <= n; row++)
    {
      index_type temp = rowPtrCsr[row];
      rowPtrCsr[row]  = last;
      last            = temp;
    }

    // Values on the host are updated now -- mark them as such!
    A_csr->setUpdated(memory::HOST);

    return 0;
  }

  /**
   * @brief Transpose a sparse CSR matrix.
   *
   * @param[in]  A - Sparse matrix
   * @param[out] At - Transposed matrix
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::transpose(matrix::Csr* A, matrix::Csr* At)
  {
    assert(A->getValues(memory::HOST) != nullptr && "Matrix A is not allocated on host.\n");
    assert(At->getValues(memory::HOST) != nullptr && "Matrix At is not allocated on host.\n");
    index_type  n        = A->getNumRows();
    index_type  m        = A->getNumColumns();
    index_type  nnz      = A->getNnz();
    index_type* rowPtrA  = A->getRowData(memory::HOST);
    index_type* colIdxA  = A->getColData(memory::HOST);
    real_type*  valuesA  = A->getValues(memory::HOST);
    index_type* rowPtrAt = At->getRowData(memory::HOST);
    index_type* colIdxAt = At->getColData(memory::HOST);
    real_type*  valuesAt = At->getValues(memory::HOST);
    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= m; ++i)
    {
      rowPtrAt[i] = 0;
    }
    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i)
    {
      colIdxAt[i] = 0;
      valuesAt[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i)
    {
      rowPtrAt[colIdxA[i]]++;
    }
    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < m; ++row)
    {
      // Store value in row pointer to temp
      index_type temp = rowPtrAt[row];

      // Copy cumulative sum to the row pointer
      rowPtrAt[row] = rowsum;

      // Update row sum
      rowsum += temp;
    }
    rowPtrAt[m] = nnz;
    for (index_type col = 0; col < n; ++col)
    {
      // Compute positions of column indices and values in CSR matrix and store them there
      // Overwrites CSR row pointers in the process
      for (index_type jj = rowPtrA[col]; jj < rowPtrA[col + 1]; jj++)
      {
        index_type row  = colIdxA[jj];
        index_type dest = rowPtrAt[row];

        colIdxAt[dest] = col;
        valuesAt[dest] = valuesA[jj];

        rowPtrAt[row]++;
      }
    }
    // Restore CSR row pointer values
    for (index_type row = 0, last = 0; row <= m; row++)
    {
      index_type temp = rowPtrAt[row];
      rowPtrAt[row]   = last;
      last            = temp;
    }
    // Values on the host are updated now -- mark them as such!
    At->setUpdated(memory::HOST);

    return 0;
  }

  /**
   * @brief Left diagonal scaling of a sparse CSR matrix
   *
   * @param[in]  diag - vector representing the diagonal matrix
   * @param[in, out]  A - Sparse CSR matrix
   *
   * @pre The diagonal vector must be of the same size as the number of rows in the matrix.
   * @pre A is unscaled and allocated
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::leftScale(vector_type* diag, matrix::Csr* A)
  {
    real_type*  diag_data = diag->getData(memory::HOST);
    index_type* rowPtrA   = A->getRowData(memory::HOST);
    real_type*  valuesA   = A->getValues(memory::HOST);

    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      for (index_type j = rowPtrA[i]; j < rowPtrA[i + 1]; ++j)
      {
        valuesA[j] *= diag_data[i];
      }
    }
    return 0;
  }

  /**
   * @brief Right diagonal scaling of a sparse CSR matrix
   *
   * @param[in]  A - Sparse CSR matrix
   * @param[in]  diag - vector representing the diagonal matrix
   *
   * @pre The diagonal vector must be of the same size as the number of columns in the matrix.
   * @pre A is unscaled
   * @post A is scaled
   * @invariant diag
   *
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::rightScale(matrix::Csr* A, vector_type* diag)
  {
    real_type*  diag_data = diag->getData(memory::HOST);
    index_type* rowPtrA   = A->getRowData(memory::HOST);
    index_type* colIdxA   = A->getColData(memory::HOST);
    real_type*  valuesA   = A->getValues(memory::HOST);

    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      for (index_type j = rowPtrA[i]; j < rowPtrA[i + 1]; ++j)
      {
        valuesA[j] *= diag_data[colIdxA[j]];
      }
    }
    return 0;
  }

  /**
   * @brief Add a constant to all nonzero values in the matrix
   *
   * @param[in, out] A - matrix
   * @param[in] alpha - constant to be added
   *
   * @return int error code, 0 if successful
   */
  int MatrixHandlerCpu::addConst(matrix::Sparse* A, real_type alpha)
  {
    real_type* values = A->getValues(memory::HOST);
    index_type nnz    = A->getNnz();
    for (index_type i = 0; i < nnz; ++i)
    {
      values[i] += alpha;
    }
    return 0;
  }

  /**
   * @brief Multiply all nonzero values of a sparse matrix by a constant
   *
   * @param[in,out] A - Sparse matrix
   * @param[in] alpha - constant to the added
   * @return 0 if successful, 1 otherwise
   */
  static int scaleConst(matrix::Sparse* A, real_type alpha)
  {
    real_type*       values = A->getValues(memory::HOST);
    const index_type nnz    = A->getNnz();
    for (index_type i = 0; i < nnz; ++i)
    {
      values[i] *= alpha;
    }
    return 0;
  }

  /**
   * @brief Update matrix with a new number of nonzero elements
   *
   * @param[in,out] A - Sparse matrix
   * @param[in] row_data - pointer to row data (array of integers)
   * @param[in] col_data - pointer to column data (array of integers)
   * @param[in] val_data - pointer to value data (array of real numbers)
   * @param[in] nnz - number of non-zer
   * @return 0 if successful, 1 otherwise
   */
  static int updateMatrix(matrix::Sparse* A, index_type* rowData, index_type* columnData, real_type* valData, index_type nnz)
  {
    if (A->destroyMatrixData(memory::HOST) != 0)
    {
      return 1;
    }
    return A->copyDataFrom(rowData, columnData, valData, nnz, memory::HOST, memory::HOST);
  }

  /**
   * @brief Add the identity to a CSR matrix.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] pattern - precalculated sparsity pattern
   * @return 0 if successful, 1 otherwise
   */
  static int addIWithPattern(matrix::Csr* A, ScaleAddBufferCpu* pattern)
  {
    std::vector<real_type> newValues(pattern->getNnz());

    index_type const* const originalRowPointers   = A->getRowData(memory::HOST);
    index_type const* const originalColumnIndices = A->getColData(memory::HOST);
    real_type const* const  originalValues        = A->getValues(memory::HOST);

    index_type newNnzCount = 0;
    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      const index_type originalRowStart = originalRowPointers[i];
      const index_type originalRowEnd   = originalRowPointers[i + 1];

      bool diagonalAdded = false;
      for (index_type j = originalRowStart; j < originalRowEnd; ++j)
      {
        if (originalColumnIndices[j] == i)
        {
          // Diagonal element found in original matrix
          newValues[newNnzCount] = originalValues[j] + 1.0;
          newNnzCount++;
          diagonalAdded = true;
        }
        else if (originalColumnIndices[j] > i && !diagonalAdded)
        {
          // Insert diagonal if not found yet
          newValues[newNnzCount] = 1.;
          newNnzCount++;
          diagonalAdded = true; // Mark as added to prevent re-insertion
          // Then add the current original element
          newValues[newNnzCount] = originalValues[j];
          newNnzCount++;
        }
        else
        {
          // Elements before diagonal, elements after diagonal and the
          // diagonal is already handled
          newValues[newNnzCount] = originalValues[j];
          newNnzCount++;
        }
      }

      // If diagonal element was not present in original row
      if (!diagonalAdded)
      {
        newValues[newNnzCount] = 1.;
        newNnzCount++;
      }
    }

    assert(newNnzCount == pattern->getNnz());
    index_type info = updateMatrix(A, pattern->getRowData(), pattern->getColumnData(), newValues.data(), pattern->getNnz());
    return info;
  }

  /**
   * @brief Add a constant to the nonzero values of a csr matrix,
   *        then add the identity matrix.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] alpha - constant to the added
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::scaleAddI(matrix::Csr* A, real_type alpha)
  {
    index_type info = scaleConst(A, alpha);
    if (info == 1)
    {
      return info;
    }

    // Reuse sparsity pattern if it is available
    if (workspace_->scaleAddISetup())
    {
      ScaleAddBufferCpu* pattern = workspace_->getScaleAddIBuffer();
      return addIWithPattern(A, pattern);
    }

    std::vector<index_type> newRowPointers(A->getNumRows() + 1);
    // At most we add one element per row/column
    index_type              maxNnzCount = A->getNnz() + A->getNumRows();
    std::vector<index_type> newColumnIndices(maxNnzCount);
    std::vector<real_type>  newValues(maxNnzCount);

    index_type const* const originalRowPointers = A->getRowData(memory::HOST);
    index_type const* const originalColIndices  = A->getColData(memory::HOST);
    real_type const* const  originalValues      = A->getValues(memory::HOST);

    index_type newNnzCount = 0;
    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      newRowPointers[i]                 = newNnzCount;
      const index_type originalRowStart = originalRowPointers[i];
      const index_type originalRowEnd   = originalRowPointers[i + 1];

      bool diagonalAdded = false;
      for (index_type j = originalRowStart; j < originalRowEnd; ++j)
      {
        if (originalColIndices[j] == i)
        {
          // Diagonal element found in original matrix
          newValues[newNnzCount]        = originalValues[j] + 1.0;
          newColumnIndices[newNnzCount] = i;
          newNnzCount++;
          diagonalAdded = true;
        }
        else if (originalColIndices[j] > i && !diagonalAdded)
        {
          // Insert diagonal if not found yet
          newValues[newNnzCount]        = 1.;
          newColumnIndices[newNnzCount] = i;
          newNnzCount++;
          diagonalAdded = true; // Mark as added to prevent re-insertion
          // Then add the current original element
          newValues[newNnzCount]        = originalValues[j];
          newColumnIndices[newNnzCount] = originalColIndices[j];
          newNnzCount++;
        }
        else
        {
          // Elements before diagonal, elements after diagonal and the
          // diagonal is already handled
          newValues[newNnzCount]        = originalValues[j];
          newColumnIndices[newNnzCount] = originalColIndices[j];
          newNnzCount++;
        }
      }

      // If diagonal element was not present in original row
      if (!diagonalAdded)
      {
        newValues[newNnzCount]        = 1.;
        newColumnIndices[newNnzCount] = i;
        newNnzCount++;
      }
    }
    newRowPointers[A->getNumRows()] = newNnzCount;
    assert(newNnzCount <= maxNnzCount);
    newColumnIndices.resize(newNnzCount);
    auto pattern = new ScaleAddBufferCpu(std::move(newRowPointers), std::move(newColumnIndices));
    // workspace_ owns pattern
    workspace_->setScaleAddIBuffer(pattern);
    updateMatrix(A, pattern->getRowData(), pattern->getColumnData(), newValues.data(), pattern->getNnz());
    return 0;
  }

  /**
   * @brief Add the identity to a CSR matrix.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in,out] B - Sparse CSR matrix
   * @param[in] pattern - precalculated sparsity pattern
   * @return 0 if successful, 1 otherwise
   */
  static int addBWithPattern(matrix::Csr* A, matrix::Csr* B, ScaleAddBufferCpu* pattern)
  {
    index_type const* const aRowPointers = A->getRowData(memory::HOST);
    index_type const* const aColIndices  = A->getColData(memory::HOST);
    real_type const* const  aValues      = A->getValues(memory::HOST);

    index_type const* const bRowPointers = B->getRowData(memory::HOST);
    index_type const* const bColIndices  = B->getColData(memory::HOST);
    real_type const* const  bValues      = B->getValues(memory::HOST);

    std::vector<real_type> newValues;
    newValues.reserve(pattern->getNnz());

    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      const index_type startA = aRowPointers[i];
      const index_type endA   = aRowPointers[i + 1];
      const index_type startB = bRowPointers[i];
      const index_type endB   = bRowPointers[i + 1];

      index_type ptrA = startA;
      index_type ptrB = startB;

      // Merge the non-zero elements for the current row
      while (ptrA < endA || ptrB < endB)
      {
        // Case 1: All remaining elements are in A
        if (ptrB >= endB)
        {
          newValues.push_back(aValues[ptrA]);
          ptrA++;
        }
        // Case 2: All remaining elements are in B
        else if (ptrA >= endA)
        {
          newValues.push_back(bValues[ptrB]);
          ptrB++;
        }
        // Case 3: Elements in both A and B. Compare column indices.
        else
        {
          if (aColIndices[ptrA] < bColIndices[ptrB])
          {
            newValues.push_back(aValues[ptrA]);
            ptrA++;
          }
          else if (bColIndices[ptrB] < aColIndices[ptrA])
          {
            newValues.push_back(bValues[ptrB]);
            ptrB++;
          }
          // Case 4: Same column index, add values
          else
          {
            real_type sum = aValues[ptrA] + bValues[ptrB];
            newValues.push_back(sum);
            ptrA++;
            ptrB++;
          }
        }
      }
    }
    assert(static_cast<index_type>(newValues.size()) == pattern->getNnz());
    return updateMatrix(A, pattern->getRowData(), pattern->getColumnData(), newValues.data(), pattern->getNnz());
  }

  /**
   * @brief Multiply csr matrix by a constant and add B.
   *
   * @param[in,out] A - Sparse CSR matrix
   * @param[in] alpha - constant to the added
   * @param[in] B - Sparse CSR matrix
   * @return 0 if successful, 1 otherwise
   */
  int MatrixHandlerCpu::scaleAddB(matrix::Csr* A, real_type alpha, matrix::Csr* B)
  {
    index_type info = scaleConst(A, alpha);
    if (info == 1)
    {
      return info;
    }

    // Reuse sparsity pattern if it is available
    if (workspace_->scaleAddBSetup())
    {
      ScaleAddBufferCpu* pattern = workspace_->getScaleAddBBuffer();
      return addBWithPattern(A, B, pattern);
    }

    index_type const* const aRowPointers = A->getRowData(memory::HOST);
    index_type const* const aColIndices  = A->getColData(memory::HOST);
    real_type const* const  aValues      = A->getValues(memory::HOST);

    index_type const* const bRowPointers = B->getRowData(memory::HOST);
    index_type const* const bColIndices  = B->getColData(memory::HOST);
    real_type const* const  bValues      = B->getValues(memory::HOST);

    std::vector<index_type> newRowPointers;
    newRowPointers.reserve(A->getNumRows() + 1);
    std::vector<index_type> newColIndices;
    std::vector<real_type>  newValues;

    newRowPointers.push_back(0); // First element is always 0
    for (index_type i = 0; i < A->getNumRows(); ++i)
    {
      const index_type startA = aRowPointers[i];
      const index_type endA   = aRowPointers[i + 1];
      const index_type startB = bRowPointers[i];
      const index_type endB   = bRowPointers[i + 1];

      index_type ptrA = startA;
      index_type ptrB = startB;

      // Merge the non-zero elements for the current row
      while (ptrA < endA || ptrB < endB)
      {
        // Case 1: All remaining elements are in A
        if (ptrB >= endB)
        {
          newColIndices.push_back(aColIndices[ptrA]);
          newValues.push_back(aValues[ptrA]);
          ptrA++;
        }
        // Case 2: All remaining elements are in B
        else if (ptrA >= endA)
        {
          newColIndices.push_back(bColIndices[ptrB]);
          newValues.push_back(bValues[ptrB]);
          ptrB++;
        }
        // Case 3: Elements in both A and B. Compare column indices.
        else
        {
          if (aColIndices[ptrA] < bColIndices[ptrB])
          {
            newColIndices.push_back(aColIndices[ptrA]);
            newValues.push_back(aValues[ptrA]);
            ptrA++;
          }
          else if (bColIndices[ptrB] < aColIndices[ptrA])
          {
            newColIndices.push_back(bColIndices[ptrB]);
            newValues.push_back(bValues[ptrB]);
            ptrB++;
          }
          // Case 4: Same column index, add values
          else
          {
            real_type sum = aValues[ptrA] + bValues[ptrB];
            newColIndices.push_back(aColIndices[ptrA]);
            newValues.push_back(sum);
            ptrA++;
            ptrB++;
          }
        }
      }
      newRowPointers.push_back(static_cast<index_type>(newValues.size()));
    }

    auto pattern = new ScaleAddBufferCpu(std::move(newRowPointers), std::move(newColIndices));
    // workspace_ owns pattern
    workspace_->setScaleAddBBuffer(pattern);
    return updateMatrix(A, pattern->getRowData(), pattern->getColumnData(), newValues.data(), pattern->getNnz());
  }
} // namespace ReSolve
