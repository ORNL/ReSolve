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

  TestOutcome matrixInfNorm(index_type N)
  {
    TestStatus status;

    matrix::Csr* A = createCsrMatrix(N);
    real_type norm;
    handler_.matrixInfNorm(A, &norm, memspace_);
    status *= (norm == 30.0); 
    
    delete A;

    return status.report(__func__);
  }

  TestOutcome matVec(index_type N)
  {
    TestStatus status;

    matrix::Csr* A = createCsrMatrix(N);
    vector::Vector x(N);
    vector::Vector y(N);
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

  TestOutcome csc2csr(index_type N, index_type M)
  {
    TestStatus status=0;
    matrix::Csc* A_csc = createRectangularCscMatrix(N, M);
    std::cout << "N: " << N << " M: " << M << "\n";
    matrix::Csr* A_csr = new matrix::Csr(M, N, 2*std::min(N,M));
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
   * @brief Create a rectangular CSR matrix
   * 
   * The sparisty structure is lower bidiagonal if N==M, with an extra entry on the first row.
   * If N>M an entry is nonzero iff i==j, or i+M==j+N 
   * if N<M an entry is nonzero iff i==j, or i+N==j+M
   * The values increase with a counter from 1.0
   * 
   * @param[in] N number of rows
   * @param[in] M number of columns
   * 
   * @return matrix::Csr* 
  */
  
  matrix::Csc* createRectangularCscMatrix(const index_type N, const index_type M)
  {
    // Allocate MXN CSR matrix with NNZ nonzeros
    index_type NNZ = 2*std::min(N,M);
    matrix::Csc* A = new matrix::Csc(M, N, NNZ); //indices are deliberately swapped so N+1 is the length of the pointer array
    A->allocateMatrixData(memory::HOST);

    index_type* colptr = A->getColData(memory::HOST);
    index_type* rowidx = A->getRowData(memory::HOST);
    real_type* val     = A->getValues( memory::HOST);

    real_type counter = 1.0;
    colptr[0] = 0;
    std::cout << "N: " << N << " M: " << M << "\n";
    if(N==M) //square case
    {
      for (index_type i=0; i < N; ++i)
      {
        colptr[i+1] = colptr[i] + 2;
        if(i==0) //first column
        {
          rowidx[colptr[i]] = i;
          val[colptr[i]] = counter++;
          rowidx[colptr[i]+1] = M/2;
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
    else if (N>M) // nonzero diagonal from main diagonal and thhe one starting in the lower right corner
    {
      for (index_type i=0; i < N; ++i)
      {
        colptr[i+1] = colptr[i];
        if(i>=N-M) //off diagonal
        {
          rowidx[colptr[i+1]] = i-N+M;
          val[colptr[i+1]] = counter++;
          colptr[i+1] ++;
        }
        if(i<M) //main diagonal
        {
          rowidx[colptr[i+1]] = i;
          val[colptr[i+1]] = counter++;
          colptr[i+1] ++;
        }
      }
    }
    else //N<M
    {
      for (index_type i=0; i < N; ++i)
      {
        colptr[i+1] = colptr[i]+2;
        rowidx[colptr[i]] = i;
        val[colptr[i]] = counter++;
        rowidx[colptr[i]+1] = i+M-N;
        val[colptr[i]+1] = counter++;
      }
    }
    A->setUpdated(memory::HOST);
    if (memspace_ == memory::DEVICE) {
      A->syncData(memspace_);
    }
    return A;        
  }

void verifyCsrMatrix(matrix::Csr* A, TestStatus& status)
{
  index_type* rowptr_csr = A->getRowData(memory::HOST);
  index_type* colidx_csr = A->getColData(memory::HOST);
  real_type* val_csr     = A->getValues( memory::HOST);
  index_type N = A->getNumColumns();
  index_type M = A->getNumRows();
  if(N==M)
  {
    for (index_type i = 0; i < M; ++i) {
      if (i==M-1) //last row should have one value
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 1);
        std::cout << "rowptr_csr[" << i+1 << "] = " << rowptr_csr[i+1] << "\n";
        status *= (colidx_csr[rowptr_csr[i]] == N-1);
        std::cout << "colidx_csr[" << rowptr_csr[i] << "] = " << colidx_csr[rowptr_csr[i]] << "\n";
        status *= (val_csr[rowptr_csr[i]] == 2.0*N);
        std::cout << "val_csr[" << rowptr_csr[i] << "] = " << val_csr[rowptr_csr[i]] << "\n";
      }
      else if(i==M/2) //this row should have 3 values
      {
        status *= (rowptr_csr[i+1] == rowptr_csr[i] + 3);
        status *= (colidx_csr[rowptr_csr[i]] == 0);
        status *= (val_csr[rowptr_csr[i]] == 2.0);
        status *= (colidx_csr[rowptr_csr[i]+1] == N/2);
        status *= (colidx_csr[rowptr_csr[i]+2] == N/2+1);
        status *= (val_csr[rowptr_csr[i]+1] == 2.0*(N/2)+2);
        status *= (val_csr[rowptr_csr[i]+2] == 2.0*(N/2)+3);
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
  else if (N>M)
  {
    index_type main_diag_ind = 0;
    index_type off_diag_ind = N-M;
    real_type main_val = 1.0;
    real_type off_val = N-M+1.0;
    for (index_type i = 0; i < M; ++i) {
      status *= (rowptr_csr[i+1] == rowptr_csr[i] + 2); // all rows should have two values
      std::cout << "rowptr_csr[" << i+1 << "] = " << rowptr_csr[i+1] << "\n";
      status *= (colidx_csr[rowptr_csr[i]] == main_diag_ind++);
      std::cout << "colidx_csr[" << rowptr_csr[i] << "] = " << colidx_csr[rowptr_csr[i]] << "main_diag_ind = " << main_diag_ind << "\n";
      status *= (colidx_csr[rowptr_csr[i]+1] == off_diag_ind++);
      std::cout << "colidx_csr[" << rowptr_csr[i]+1 << "] = " << colidx_csr[rowptr_csr[i]+1] << "off_diag_ind = " << off_diag_ind << "\n";
      status *= (val_csr[rowptr_csr[i]] == main_val++);
      std::cout << "val_csr[" << rowptr_csr[i] << "] = " << val_csr[rowptr_csr[i]] << "main_val = " << main_val << "\n";
      status *= (val_csr[rowptr_csr[i]+1] == off_val++);
      std::cout << "val_csr[" << rowptr_csr[i]+1 << "] = " << val_csr[rowptr_csr[i]+1] << "off_val = " << off_val << "\n";
      if (i>=N-M-1)
      {
        main_val++;
      }
      if(i<2*M-N)
      {
        off_val++;
      }
      
    }
  }
}

  matrix::Csr* createCsrMatrix(const index_type N)
  {
    std::vector<real_type> r1 = {1., 5., 7., 8., 3., 2., 4.}; // sum 30
    std::vector<real_type> r2 = {1., 3., 2., 2., 1., 6., 7., 3., 2., 3.}; // sum 30
    std::vector<real_type> r3 = {11., 15., 4.}; // sum 30
    std::vector<real_type> r4 = {1., 1., 5., 1., 9., 2., 1., 2., 3., 2., 3.}; // sum 30
    std::vector<real_type> r5 = {6., 5., 7., 3., 2., 5., 2.}; // sum 30

    const std::vector<std::vector<real_type> > data = {r1, r2, r3, r4, r5};

    // std::cout << N << "\n";

    // First compute number of nonzeros
    index_type NNZ = 0;
    for (index_type i = 0; i < N; ++i)
    {
      size_t reminder = static_cast<size_t>(i%5);
      NNZ += static_cast<index_type>(data[reminder].size());
    }

    // Allocate NxN CSR matrix with NNZ nonzeros
    matrix::Csr* A = new matrix::Csr(N, N, NNZ);
    A->allocateMatrixData(memory::HOST);

    index_type* rowptr = A->getRowData(memory::HOST);
    index_type* colidx = A->getColData(memory::HOST);
    real_type* val     = A->getValues( memory::HOST); 

    // Populate CSR matrix using same row pattern as for NNZ calculation
    rowptr[0] = 0;
    for (index_type i=0; i < N; ++i)
    {
      size_t reminder = static_cast<size_t>(i%5);
      const std::vector<real_type>& row_sample = data[reminder];
      index_type nnz_per_row = static_cast<index_type>(row_sample.size());

      rowptr[i+1] = rowptr[i] + nnz_per_row;
      for (index_type j = rowptr[i]; j < rowptr[i+1]; ++j)
      {
        colidx[j] = (j - rowptr[i]) * N/nnz_per_row + (N%(N/nnz_per_row));
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
