#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <resolve/matrix/Csr.hpp>
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
      std::cout << "Oups we have an issue \n";
    y.allocate(memspace_);

    x.setToConst(1.0, memspace_);
    y.setToConst(1.0, memspace_);

    real_type alpha = 2.0/30.0;
    real_type beta  = 2.0;
    handler_.setValuesChanged(true, memspace_);
    handler_.matvec(A, &x, &y, &alpha, &beta, memspace_);

    status *= verifyAnswer(y, 4.0);

    delete A;

    return status.report(__func__);
  }

  TestOutcome csc2csr(index_type N, index_type M)
  {
    TestStatus status=0;
    matrix::Csr* A = createRectangularCsrMatrix(N, M);
    matrix::Csc* A_csc = new matrix::Csc(N, M, 2*std::min(N,M));
    A_csc->allocateMatrixData(memory::HOST);

    handler_.csc2csr(A, A_csc, memspace_);

    status *= (A->getNumRows() == A_csc->getNumRows());
    status *= (A->getNumColumns() == A_csc->getNumColumns());

    // verify the second and second to last entry in the column and value arrays



    delete A;
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
      // std::cout << x.getData(memory::HOST)[i] << "\n";
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
  
  matrix::Csr* createRectangularCsrMatrix(const index_type N, const index_type M)
  {
    // Allocate NxM CSR matrix with NNZ nonzeros
    index_type NNZ = 2*std::min(N,M);
    matrix::Csr* A = new matrix::Csr(N, M, NNZ);
    A->allocateMatrixData(memory::HOST);

    index_type* rowptr = A->getRowData(memory::HOST);
    index_type* colidx = A->getColData(memory::HOST);
    real_type* val     = A->getValues( memory::HOST);

    real_type counter = 1.0;

    if(N==M) //square case
    {
      rowptr[0] = 0;
      for (index_type i=0; i < N; ++i)
      {
        rowptr[i+1] = rowptr[i] + 2;
        if(i==0) //first row
        {
          colidx[rowptr[i]] = i;
          val[rowptr[i]] = counter++;
          colidx[rowptr[i]+1] = M/2;
          val[rowptr[i]+1] = counter++;
        }
        else
        {
          colidx[rowptr[i]] = i-1;
          val[rowptr[i]] = counter++;
          colidx[rowptr[i]+1] = i;
          val[rowptr[i]+1] = counter++;
        }
      }
    }
    else if (N>M)
    {
      rowptr[0] = 0;
      for (index_type i=0; i < N; ++i)
      {
        rowptr[i+1] = rowptr[i];
        if(i>=N-M) //off diagonal
        {
          colidx[rowptr[i+1]] = i-N+M;
          val[rowptr[i+1]] = counter++;
          rowptr[i+1] ++;
        }
        if(i<M) //main diagonal
        {
          colidx[rowptr[i+1]] = i;
          val[rowptr[i+1]] = counter++;
          rowptr[i+1] ++;
        }
      }
    }
    else //N<M
    {
      rowptr[0] = 0;
      for (index_type i=0; i < N; ++i)
      {
        rowptr[i+1] = rowptr[i]+2;
        colidx[rowptr[i]] = i;
        val[rowptr[i]] = counter++;
        colidx[rowptr[i]+1] = i+M-N;
        val[rowptr[i]+1] = counter++;
      }
    }
    A->setUpdated(memory::HOST);

    if (memspace_ == memory::DEVICE) {
      A->syncData(memspace_);
    }

    return A;        
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
