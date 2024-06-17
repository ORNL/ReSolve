#include <algorithm>
#include <cassert>
#include <memory>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include "MatrixHandlerCpu.hpp"

namespace ReSolve {
  // Create a shortcut name for Logger static class
  using out = io::Logger;

  MatrixHandlerCpu::MatrixHandlerCpu()
  {
  }

  MatrixHandlerCpu::~MatrixHandlerCpu()
  {
  }

  MatrixHandlerCpu::MatrixHandlerCpu(LinAlgWorkspaceCpu* new_workspace)
  {
    workspace_ = new_workspace;
  }

  void MatrixHandlerCpu::setValuesChanged(bool values_changed)
  {
    values_changed_ = values_changed;
  }


  /**
   * @brief result := alpha * A * x + beta * result
   */
  int MatrixHandlerCpu::matvec(matrix::Sparse* Ageneric, 
                               vector_type* vec_x, 
                               vector_type* vec_result, 
                               const real_type* alpha, 
                               const real_type* beta,
                               std::string matrixFormat) 
  {
    using namespace constants;
    // int error_sum = 0;
    if (matrixFormat == "csr") {
      matrix::Csr* A = (matrix::Csr*) Ageneric;
      index_type* ia = A->getRowData(memory::HOST);
      index_type* ja = A->getColData(memory::HOST);
      real_type*   a = A->getValues( memory::HOST);

      real_type* x_data      = vec_x->getData(memory::HOST);
      real_type* result_data = vec_result->getData(memory::HOST);
      real_type sum;
      real_type y;
      real_type t;
      real_type c;

      //Kahan algorithm for stability; Kahan-Babushka version didnt make a difference   
      for (int i = 0; i < A->getNumRows(); ++i) {
        sum = 0.0;
        c = 0.0;
        for (int j = ia[i]; j < ia[i+1]; ++j) { 
          y =  ( a[j] * x_data[ja[j]]) - c;
          t = sum + y;
          c = (t - sum) - y;
          sum = t;
          //  sum += ( a[j] * x_data[ja[j]]);
        }
        sum *= (*alpha);
        result_data[i] = result_data[i]*(*beta) + sum;
      } 
      vec_result->setDataUpdated(memory::HOST);
      return 0;
    } else {
      out::error() << "MatVec not implemented (yet) for " 
                   << matrixFormat << " matrix format." << std::endl;
      return 1;
    }
  }

  int MatrixHandlerCpu::matrixInfNorm(matrix::Sparse* A, real_type* norm)
  {
    real_type sum, nrm;
    index_type i, j;

    for (i = 0; i < A->getNumRows(); ++i) {
      sum = 0.0; 
      for (j  = A->getRowData(memory::HOST)[i]; j < A->getRowData(memory::HOST)[i+1]; ++j) {
        sum += std::abs(A->getValues(memory::HOST)[j]);
      }
      if (i == 0) {
        nrm = sum;
      } else {
        if (sum > nrm)
        {
          nrm = sum;
        } 
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
   */
  int MatrixHandlerCpu::csc2csr(matrix::Csc* A_csc, matrix::Csr* A_csr)
  {
    // int error_sum = 0; TODO: Collect error output!
    assert(A_csc->getNnz() == A_csr->getNnz());
    assert(A_csc->getNumRows() == A_csr->getNumColumns());
    assert(A_csr->getNumRows() == A_csc->getNumColumns());

    index_type nnz = A_csc->getNnz();
    index_type n   = A_csc->getNumColumns();

    index_type* rowIdxCsc = A_csc->getRowData(memory::HOST);
    index_type* colPtrCsc = A_csc->getColData(memory::HOST);
    real_type*  valuesCsc = A_csc->getValues( memory::HOST);

    index_type* rowPtrCsr = A_csr->getRowData(memory::HOST);
    index_type* colIdxCsr = A_csr->getColData(memory::HOST);
    real_type*  valuesCsr = A_csr->getValues( memory::HOST);

    // Set all CSR row pointers to zero
    for (index_type i = 0; i <= n; ++i) {
      rowPtrCsr[i] = 0;
    }

    // Set all CSR values and column indices to zero
    for (index_type i = 0; i < nnz; ++i) {
      colIdxCsr[i] = 0;
      valuesCsr[i] = 0.0;
    }

    // Compute number of entries per row
    for (index_type i = 0; i < nnz; ++i) {
      rowPtrCsr[rowIdxCsc[i]]++;
    }

    // Compute cumualtive sum of nnz per row
    for (index_type row = 0, rowsum = 0; row < n; ++row)
    {
      // Store value in row pointer to temp
      index_type temp  = rowPtrCsr[row];

      // Copy cumulative sum to the row pointer
      rowPtrCsr[row] = rowsum;

      // Update row sum
      rowsum += temp;
    }
    rowPtrCsr[n] = nnz;

    for (index_type col = 0; col < n; ++col)
    {
      // Compute positions of column indices and values in CSR matrix and store them there
      // Overwrites CSR row pointers in the process
      for (index_type jj = colPtrCsc[col]; jj < colPtrCsc[col+1]; jj++)
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
        index_type temp  = rowPtrCsr[row];
        rowPtrCsr[row] = last;
        last    = temp;
    }

    return 0;
  }

} // namespace ReSolve
