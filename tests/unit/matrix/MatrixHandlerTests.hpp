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

  TestOutcome matrixOneNorm()
  {
    TestStatus status;
    status.skipTest();
    
    return status.report(__func__);
  }

  TestOutcome matVec(index_type N)
  {
    TestStatus status;

    ReSolve::MatrixHandler* handler = createMatrixHandler();

    matrix::Csr* A = createCsrMatrix(N, memspace_);
    vector::Vector x(N);
    vector::Vector y(N);
    x.allocate(memspace_);
    y.allocate(memspace_);

    x.setToConst(1.0, memspace_);
    y.setToConst(1.0, memspace_);

    real_type alpha = 2.0/30.0;
    real_type beta  = 2.0;
    handler->setValuesChanged(true, memspace_);
    handler->matvec(A, &x, &y, &alpha, &beta, "csr", memspace_);

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
      LinAlgWorkspaceCpu* workpsace = new LinAlgWorkspaceCpu();
      return new MatrixHandler(workpsace);
    } else if (memspace_ == "cuda") {
      LinAlgWorkspaceCUDA* workspace = new LinAlgWorkspaceCUDA();
      workspace->initializeHandles();
      return new MatrixHandler(workspace);
    } else {
      std::cout << "Invalid memory space " << memspace_ << "\n";
    }
    return nullptr;
  }

  bool verifyAnswer(vector::Vector& x, real_type answer, std::string memspace)
  {
    bool status = true;
    if (memspace != "cpu") {
      x.copyData(memspace, "cpu");
    }

    for (index_type i = 0; i < x.getSize(); ++i) {
      // std::cout << x.getData("cpu")[i] << "\n";
      if (!isEqual(x.getData("cpu")[i], answer)) {
        status = false;
        std::cout << "Solution vector element x[" << i << "] = " << x.getData("cpu")[i]
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

    index_type NNZ = 0;
    for (index_type i = 0; i < N; ++i)
    {
      NNZ += static_cast<index_type>(data[i%5].size());
    }
    // std::cout << NNZ << "\n";

    matrix::Csr* A = new matrix::Csr(N, N, NNZ);
    A->allocateMatrixData("cpu");

    index_type* rowptr = A->getRowData("cpu");
    index_type* colidx = A->getColData("cpu");
    real_type* val     = A->getValues("cpu"); 

    rowptr[0] = 0;
    index_type i = 0;
    for (i=0; i < N; ++i)
    {
      const std::vector<real_type>& row_sample = data[i%5];
      index_type nnz_per_row = static_cast<index_type>(row_sample.size());
      // std::cout << nnz_per_row << "\n";

      rowptr[i+1] = rowptr[i] + nnz_per_row;
      for (index_type j = rowptr[i]; j < rowptr[i+1]; ++j)
      {
        colidx[j] = (j - rowptr[i]) * N/nnz_per_row + (N%(N/nnz_per_row));
        // evenly distribute nonzeros ^^^^             ^^^^^^^^ perturb offset
        val[j] = row_sample[j - rowptr[i]];
        // std::cout << i << " " << colidx[j] << "  " << val[j] << "\n";
      }
    }
    A->setUpdated("cpu");
    // std::cout << rowptr[i] << "\n";

    if (memspace == "cuda") {
      A->copyData(memspace);
    }

    return A;
  }
}; // class MatrixHandlerTests

}} // namespace ReSolve::tests
