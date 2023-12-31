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
  MatrixHandlerTests(std::string memspace) : memspace_(memspace) 
  {}
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
    ReSolve::memory::MemorySpace ms;
    if (memspace_ == "cpu")
      ms = memory::HOST;
    else
      ms = memory::DEVICE;

    ReSolve::MatrixHandler* handler = createMatrixHandler();

    matrix::Csr* A = createCsrMatrix(N, memspace_);
    real_type norm;
    handler->matrixInfNorm(A, &norm, ms);
    status *= (norm == 30.0); 
    
    delete handler;
    delete A;
    return status.report(__func__);
  }

  TestOutcome matVec(index_type N)
  {
    TestStatus status;
    ReSolve::memory::MemorySpace ms;
    if (memspace_ == "cpu")
      ms = memory::HOST;
    else
      ms = memory::DEVICE;

    ReSolve::MatrixHandler* handler = createMatrixHandler();

    matrix::Csr* A = createCsrMatrix(N, memspace_);
    vector::Vector x(N);
    vector::Vector y(N);
    x.allocate(ms);
    if (x.getData(ms) == NULL) printf("oups we have an issue \n");
    y.allocate(ms);

    x.setToConst(1.0, ms);
    y.setToConst(1.0, ms);

    real_type alpha = 2.0/30.0;
    real_type beta  = 2.0;
    handler->setValuesChanged(true, ms);
    handler->matvec(A, &x, &y, &alpha, &beta, "csr", ms);

    status *= verifyAnswer(y, 4.0, memspace_);

    delete handler;
    delete A;

    return status.report(__func__);
  }

private:
  std::string memspace_{"cpu"};

  ReSolve::MatrixHandler* createMatrixHandler()
  {
    if (memspace_ == "cpu") {
      LinAlgWorkspaceCpu* workspace = new LinAlgWorkspaceCpu();
      return new MatrixHandler(workspace);
#ifdef RESOLVE_USE_CUDA
    } else if (memspace_ == "cuda") {
      LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
      workspace->initializeHandles();
      return new MatrixHandler(workspace);
#endif
#ifdef RESOLVE_USE_HIP
    } else if (memspace_ == "hip") {
      LinAlgWorkspaceHIP* workspace = new LinAlgWorkspaceHIP();
      workspace->initializeHandles();
      return new MatrixHandler(workspace);
#endif
    } else {
      std::cout << "ReSolve not built with support for memory space " << memspace_ << "\n";
    }
    return nullptr;
  }

  bool verifyAnswer(vector::Vector& x, real_type answer, std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      x.copyData(memory::DEVICE, memory::HOST);
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

  matrix::Csr* createCsrMatrix(const index_type N, std::string memspace)
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

    if ((memspace == "cuda") || (memspace == "hip")) {
      A->copyData(memory::DEVICE);
    }

    return A;
  }
}; // class MatrixHandlerTests

}} // namespace ReSolve::tests
