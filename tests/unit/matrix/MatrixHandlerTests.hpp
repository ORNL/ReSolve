#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve { namespace tests {

/**
 * @class Unit tests for matrix handler class
 */
class MatrixHandlerTests : TestBase
{
public:
  MatrixHandlerTests(ReSolve::MatrixHandler& handler) : handler_(handler) 
  {
    if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled()) {
      memspace_ = memory::DEVICE;
    } else {
      memspace_ = memory::HOST;
    }
  }

  virtual ~MatrixHandlerTests()
  {}

  TestOutcome matrixHandlerConstructor()
  {
    TestStatus status;
    status.skipTest();
    
    return status.report(__func__);
  }

  TestOutcome matrixInfNorm(index_type n)
  {
    TestStatus status;

    matrix::Csr* A = createCsrMatrix(n);
    real_type norm;
    handler_.matrixInfNorm(A, &norm, memspace_);
    status *= (norm == 30.0); 
    
    delete A;

    return status.report(__func__);
  }

  TestOutcome matVec(index_type n)
  {
    TestStatus status;

    matrix::Csr* A = createCsrMatrix(n);
    vector::Vector x(n);
    vector::Vector y(n);
    x.allocate(memspace_);
    if (x.getData(memspace_) == NULL) 
      std::cout << "The memory space was not allocated \n" << std::endl;
    y.allocate(memspace_);

    x.setToConst(1.0, memspace_);
    y.setToConst(1.0, memspace_);

    real_type alpha = 2.0/30.0;
    real_type beta  = 2.0;
    handler_.setValuesChanged(true, memspace_);
    handler_.matvec(A, &x, &y, &alpha, &beta, memspace_);

    status *= verifyAnswer(y, 4.0);

    //delete A;

    return status.report(__func__);
  }

  TestOutcome csc2csr(index_type n, index_type m)
  {
    TestStatus status=0;
    matrix::Csc* A_csc = createRectangularCscMatrix(n, m);
    // printMatrix(A_csc, "CSC matrix");
    std::cout << "n: " << n << " m: " << m << "\n";

    matrix::Csr* A_csr = new matrix::Csr(m, n, 2*std::min(n,m));
    A_csr->allocateMatrixData(memory::HOST);

    handler_.csc2csr(A_csc, A_csr, memspace_);

    status *= (A_csr->getNumRows() == A_csc->getNumRows());
    status *= (A_csr->getNumColumns() == A_csc->getNumColumns());
    status *= (A_csr->getNnz() == A_csc->getNnz());

    if (memspace_ == memory::DEVICE) {
      //update the data on the device
      A_csr->setUpdated(memory::DEVICE);
      A_csr->syncData(memory::HOST);
    }

    index_type* rowptr_csr = A_csr->getRowData(memory::HOST);
    index_type* colidx_csr = A_csr->getColData(memory::HOST);
    real_type* val_csr     = A_csr->getValues( memory::HOST);

    verifyCsrMatrix(A_csr, status);

    delete A_csr;
    delete A_csc;

    return status.report(__func__);
  }

private:
  ReSolve::MatrixHandler& handler_;
  memory::MemorySpace memspace_{memory::HOST};

  bool verifyAnswer(vector::Vector& x, real_type answer)
  {
    bool status = true;
    if (memspace_ == memory::DEVICE) {
      x.syncData(memory::HOST);
    }

    for (index_type i = 0; i < x.getSize(); ++i) {
      if (!isEqual(x.getData(memory::HOST)[i], answer)) {
        status = false;
        std::cout << "Solution vector element x[" << i << "] = " << x.getData(memory::HOST)[i]
                  << ", expected: " << answer << "\n";
        break; 
      }
    }
    return status;
  }
  /** 
   * @brief Create a rectangular CSC matrix with preset sparsity structure
   * 
   * The sparisty structure is upper bidiagonal if n==m, with an extra entry in the first column
   * If n>m an entry is nonzero iff i==j, or i+m==j+n 
   * if n<m an entry is nonzero iff i==j, or i+n==j+m
   * The values increase with a counter from 1.0 in column major order.
   * 
   * @param[in] n number of columns
   * @param[in] m number of rows
   * 
   * @return matrix::Csr* 
  */
  
  matrix::Csc* createRectangularCscMatrix(const index_type n, const index_type m)
  {
    // Allocate mxn CSR matrix with nnz nonzeros
    index_type nnz = 2*std::min(n,m);
    matrix::Csc* A = new matrix::Csc(m, n, nnz); //indices are deliberately swapped so n+1 is the length of the pointer array
    A->allocateMatrixData(memory::HOST);

    index_type* colptr = A->getColData(memory::HOST);
    index_type* rowidx = A->getRowData(memory::HOST);
    real_type* val     = A->getValues( memory::HOST);

    real_type counter = 1.0;
    colptr[0] = 0;
    if(n==m) //square case
    {
      for (index_type i=0; i < n; ++i)
      {
        colptr[i+1] = colptr[i] + 2;
        if(i==0) //first column
        {
          rowidx[colptr[i]] = i;
          val[colptr[i]] = counter++;
          rowidx[colptr[i]+1] = m/2;
          val[colptr[i]+1] = counter++;
        }
        else
        {
          rowidx[colptr[i]] = i-1;
          val[colptr[i]] = counter++;
          rowidx[colptr[i]+1] = i;
          val[colptr[i]+1] = counter++;
        }
      }
    }
    else if (n>m) // nonzero diagonal from main diagonal and thhe one starting in the lower right corner
    {
      for (index_type i=0; i < n; ++i)
      {
        colptr[i+1] = colptr[i];
        if(i>=n-m) //off diagonal
        {
          rowidx[colptr[i+1]] = i-n+m;
          val[colptr[i+1]] = counter++;
          colptr[i+1] ++;
        }
        if(i<m) //main diagonal
        {
          rowidx[colptr[i+1]] = i;
          val[colptr[i+1]] = counter++;
          colptr[i+1] ++;
        }
      }
    }
    else //n<m
    {
      for (index_type i=0; i < n; ++i)
      {
        colptr[i+1] = colptr[i]+2;
        rowidx[colptr[i]] = i;
        val[colptr[i]] = counter++;
        rowidx[colptr[i]+1] = i+m-n;
        val[colptr[i]+1] = counter++;
      }
    }
    A->setUpdated(memory::HOST);
    if (memspace_ == memory::DEVICE) {
      A->syncData(memspace_);
    }
    return A;        
  }

/*
  * @brief Verify structure of a CSR matrix with preset pattern.
  * 
  * The sparsity structure corresponds to the CSR representation of a rectangular matrix
  * created by createRectangularCscMatrix.
  * The sparisty structure is upper bidiagonal if n==m, 
  * with an extra entry in the first column.
  * If n>m an entry is nonzero iff i==j, or i+m==j+n 
  * if n<m an entry is nonzero iff i==j, or i+n==j+m
  * The values increase with a counter from 1.0 in column major order.
  * 
  * @param[in] A matrix::Csr* pointer to the matrix to be verified
  * @param[out] status TestStatus& reference to the status of the test
  */


void verifyCsrMatrix(matrix::Csr* A, TestStatus& status)
{
  index_type* rowptr_csr = A->getRowData(memory::HOST);
  index_type* colidx_csr = A->getColData(memory::HOST);
  real_type* val_csr     = A->getValues( memory::HOST);
  index_type n = A->getNumColumns();
  index_type m = A->getNumRows();
  if(n==m)
  {
    for (index_type i = 0; i < m; ++i) {
      if (i==m-1) //last row should have one value
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 1);
        status *= (colidx_csr[rowptr_csr[i]] == n-1);
        status *= (val_csr[rowptr_csr[i]] == 2.0*n);
      }
      else if(i==m/2) //this row should have 3 values
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 3);
        status *= (colidx_csr[rowptr_csr[i]] == 0);
        status *= (val_csr[rowptr_csr[i]] == 2.0);
        status *= (colidx_csr[rowptr_csr[i]+1] == n/2);
        status *= (colidx_csr[rowptr_csr[i]+2] == n/2+1);
        status *= (val_csr[rowptr_csr[i]+1] == 2.0*(n/2)+2);
        status *= (val_csr[rowptr_csr[i]+2] == 2.0*(n/2)+3);
      }
      else // all other rows have two values
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 2);
        status *= (colidx_csr[rowptr_csr[i]] == i);
        status *= (colidx_csr[rowptr_csr[i]+1] == i+1);
        if(i==0)
        {
          status *= (val_csr[rowptr_csr[i]] == 1.0);
          status *= (val_csr[rowptr_csr[i]+1] == 3.0);
        }
        else
        {
          status *= (val_csr[rowptr_csr[i]] == 2.0*(i+1));
          status *= (val_csr[rowptr_csr[i]+1] == 2.0*(i+1)+1.0);
        }

      }
    }
  }
  else if (n>m)
  {
    index_type main_diag_ind = 0;
    index_type off_diag_ind = n-m;
    real_type main_val = 1.0;
    real_type off_val = n-m+1.0;
    for (index_type i = 0; i < m; ++i) {
      status *= (rowptr_csr[i+1] == rowptr_csr[i] + 2); // all rows should have two values
      status *= (colidx_csr[rowptr_csr[i]] == main_diag_ind++);
      status *= (colidx_csr[rowptr_csr[i]+1] == off_diag_ind++);
      status *= (val_csr[rowptr_csr[i]] == main_val++);
      status *= (val_csr[rowptr_csr[i]+1] == off_val++);
      if (i>=n-m-1)
      {
        main_val++;
      }
      if(i<2*m-n)
      {
        off_val++;
      }
      
    }
  }
  else // m>n
  {
    real_type main_val = 1.0;
    real_type off_val = 2.0;
    for (index_type i = 0; i < m; ++i) {
      if(i<n && i<m-n) // only main diagonal
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 1); 
        status *= (colidx_csr[rowptr_csr[i]] == i);
        status *= (val_csr[rowptr_csr[i]] == main_val);
        main_val+=2.0;
      }
      else if (i<n && i>=m-n)
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 2); 
        status *= (colidx_csr[rowptr_csr[i]+1] == i);
        status *= (colidx_csr[rowptr_csr[i]] == i+n-m);
        status *= (val_csr[rowptr_csr[i]+1] == main_val);
        status *= (val_csr[rowptr_csr[i]] == off_val);
        main_val+=2.0;
        off_val+=2.0;
      }
      else
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 1); 
        status *= (colidx_csr[rowptr_csr[i]] == i+n-m);
        status *= (val_csr[rowptr_csr[i]] == off_val);
        off_val+=2.0;
      }
      
    }

  }
}

  /**
   * @brief Create a CSR matrix with preset sparsity structure
   * 
   * The sparisty structure is such that each row has a different number of nonzeros
   * The values are chosen so that the sum of each row is 30
   * 
   * @param[in] n number of rows and columns
   * 
   * @return matrix::Csr* 
   */

matrix::Csr* createCsrMatrix(const index_type n)
{
    std::vector<real_type> r1 = {1., 5., 7., 8., 3., 2., 4.}; // sum 30
    std::vector<real_type> r2 = {1., 3., 2., 2., 1., 6., 7., 3., 2., 3.}; // sum 30
    std::vector<real_type> r3 = {11., 15., 4.}; // sum 30
    std::vector<real_type> r4 = {1., 1., 5., 1., 9., 2., 1., 2., 3., 2., 3.}; // sum 30
    std::vector<real_type> r5 = {6., 5., 7., 3., 2., 5., 2.}; // sum 30

    const std::vector<std::vector<real_type> > data = {r1, r2, r3, r4, r5};

    // std::cout << n << "\n";

    // First compute number of nonzeros
    index_type nnz = 0;
    for (index_type i = 0; i < n; ++i)
    {
      size_t reminder = static_cast<size_t>(i%5);
      nnz += static_cast<index_type>(data[reminder].size());
    }

    // Allocate nxn CSR matrix with nnz nonzeros
    matrix::Csr* A = new matrix::Csr(n, n, nnz);
    A->allocateMatrixData(memory::HOST);

    index_type* rowptr = A->getRowData(memory::HOST);
    index_type* colidx = A->getColData(memory::HOST);
    real_type* val     = A->getValues( memory::HOST); 

    // Populate CSR matrix using same row pattern as for nnz calculation
    rowptr[0] = 0;
    for (index_type i=0; i < n; ++i)
    {
      size_t reminder = static_cast<size_t>(i%5);
      const std::vector<real_type>& row_sample = data[reminder];
      index_type nnz_per_row = static_cast<index_type>(row_sample.size());

      rowptr[i+1] = rowptr[i] + nnz_per_row;
      for (index_type j = rowptr[i]; j < rowptr[i+1]; ++j)
      {
        colidx[j] = (j - rowptr[i]) * n/nnz_per_row + (n%(n/nnz_per_row));
        // evenly distribute nonzeros ^^^^             ^^^^^^^^ perturb offset
        val[j] = row_sample[static_cast<size_t>(j - rowptr[i])];
      }
    }
    A->setUpdated(memory::HOST);

    if (memspace_ == memory::DEVICE) {
      A->syncData(memspace_);
    }

    return A;
}
}; // class MatrixHandlerTests

}} // namespace ReSolve::tests
