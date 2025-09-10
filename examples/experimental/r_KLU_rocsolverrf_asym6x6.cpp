/*
This is a simple example demonstrating the use of ReSolve's direct solvers KLU with RocSolverRf on the same system.
Previously, this example was flagged as failing in issue #332.
*/

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "examples/ExampleHelper.hpp"
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspaceHIP.hpp>

int main()
{
  std::cout << "Simple ReSolve Example (following gpuRefactor pattern)\n";
  std::cout << "=====================================================\n";

  // Matrix data in CSC format (converted from 1-based to 0-based indexing)
  // Julia CSC data: colptr, rowval, nzval
  std::vector<int>    csc_col_ptr = {0, 1, 2, 3, 6, 8, 10};         // Julia colptr - 1
  std::vector<int>    csc_row_ind = {0, 1, 2, 0, 1, 3, 0, 4, 1, 5}; // Julia rowval - 1
  std::vector<double> csc_values  = {
      0.00141521219668486,
      0.6984313028075056,
      -10.0,
      -0.8307014463420237,
      0.1846074461471301,
      -10.0,
      -0.8408959803799315,
      -10.0,
      -0.835721417925707,
      -10.0};
  using namespace ReSolve::examples;
  // Convert CSC to CSR manually for this example.
  int n   = 6;                                   // 6x6 matrix
  int nnz = static_cast<int>(csc_values.size()); // 10 non-zeros

  std::cout << "Matrix size: " << n << "x" << n << "\n";
  std::cout << "Number of non-zeros: " << nnz << "\n";

  // Convert CSC to CSR.
  std::vector<int>    csr_row_ptr(static_cast<size_t>(n + 1), 0);
  std::vector<int>    csr_col_ind(static_cast<size_t>(nnz));
  std::vector<double> csr_values(static_cast<size_t>(nnz));

  // Count entries per row.
  for (size_t i = 0; i < static_cast<size_t>(nnz); ++i)
  {
    csr_row_ptr[static_cast<size_t>(csc_row_ind[i] + 1)]++;
  }

  // Convert counts to pointers
  for (size_t i = 1; i <= static_cast<size_t>(n); ++i)
  {
    csr_row_ptr[i] += csr_row_ptr[i - 1];
  }

  // Fill CSR arrays
  std::vector<int> temp_row_ptr = csr_row_ptr;
  for (size_t col = 0; col < static_cast<size_t>(n); ++col)
  {
    for (size_t idx = static_cast<size_t>(csc_col_ptr[col]); idx < static_cast<size_t>(csc_col_ptr[col + 1]); ++idx)
    {
      size_t row       = static_cast<size_t>(csc_row_ind[idx]);
      size_t pos       = static_cast<size_t>(temp_row_ptr[row]++);
      csr_col_ind[pos] = static_cast<int>(col);
      csr_values[pos]  = csc_values[idx];
    }
  }

  std::cout << "Matrix data converted to CSR format\n";

  // Create workspace
  ReSolve::LinAlgWorkspaceHIP workspace;
  workspace.initializeHandles();
  ExampleHelper<ReSolve::LinAlgWorkspaceHIP> helper(workspace);

  // Create matrix and vector handlers
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);

  // Direct solvers instantiation
  ReSolve::LinSolverDirectKLU         KLU;
  ReSolve::LinSolverDirectRocSolverRf Rf(&workspace);

  // Create CSR matrix
  ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, n, nnz, false, false);
  A->allocateMatrixData(ReSolve::memory::HOST);
  A->copyDataFrom(csr_row_ptr.data(), csr_col_ind.data(), csr_values.data(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  A->allocateMatrixData(ReSolve::memory::DEVICE);
  A->syncData(ReSolve::memory::DEVICE);

  std::cout << "CSR matrix created and synced to device\n";

  // A->print(std::cout, 0);

  // Create vectors
  ReSolve::vector::Vector* vec_rhs = new ReSolve::vector::Vector(n);
  ReSolve::vector::Vector* vec_x   = new ReSolve::vector::Vector(n);

  vec_rhs->allocate(ReSolve::memory::HOST);
  vec_rhs->allocate(ReSolve::memory::DEVICE);
  vec_x->allocate(ReSolve::memory::HOST);
  vec_x->allocate(ReSolve::memory::DEVICE);

  // Set right-hand side vector
  std::vector<double> rhs_data = {
      0.04204480190421101,
      0.1868729984154292,
      -0.1581138884334949,
      0.04939382906591484,
      -1.0,
      0.1118034015563139};
  vec_rhs->copyDataFrom(rhs_data.data(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->syncData(ReSolve::memory::DEVICE);

  std::cout << "Right-hand side vector set\n";

  // Setup KLU solver
  KLU.setup(A);
  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);

  // Analysis (symbolic factorization)
  int status = KLU.analyze();
  std::cout << "KLU analysis status: " << status << std::endl;
  if (status != 0)
  {
    std::cerr << "KLU analysis failed!" << std::endl;
    return 1;
  }

  // Numeric factorization
  status = KLU.factorize();
  std::cout << "KLU factorization status: " << status << std::endl;
  if (status != 0)
  {
    std::cerr << "KLU factorization failed!" << std::endl;
    return 1;
  }

  // Triangular solve
  status = KLU.solve(vec_rhs, vec_x);
  std::cout << "KLU solve status: " << status << std::endl;
  if (status != 0)
  {
    std::cerr << "KLU solve failed!" << std::endl;
    return 1;
  }

  std::cout << "System solved successfully with KLU\n";

  // Get solution
  double* solution = vec_x->getData(ReSolve::memory::HOST);

  std::cout << "Solution vector:\n";
  std::cout << "[";
  for (int i = 0; i < n; ++i)
  {
    std::cout << solution[i] << (i < n - 1 ? ", " : "");
  }
  std::cout << "]" << std::endl;

  helper.resetSystem(A, vec_rhs, vec_x);
  helper.printShortSummary();
  ReSolve::matrix::Csr* L = (ReSolve::matrix::Csr*) KLU.getLFactorCsr();
  ReSolve::matrix::Csr* U = (ReSolve::matrix::Csr*) KLU.getUFactorCsr();
  if (L == nullptr || U == nullptr)
  {
    std::cout << "Factor extraction from KLU failed!\n";
  }
  else
  {
    std::cout << "L and U factors extracted successfully\n";
    // Print L and U factors
    std::cout << "L factor:\n";
    L->print(std::cout, 0);
    std::cout << "U factor:\n";
    U->print(std::cout, 0);

    ReSolve::index_type* P          = KLU.getPOrdering();
    ReSolve::index_type* Q          = KLU.getQOrdering();
    double*              rhs_before = vec_rhs->getData(ReSolve::memory::HOST);
    std::cout << "[";
    for (int i = 0; i < n; ++i)
    {
      std::cout << rhs_before[i] << (i < n - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    Rf.setupCsr(A, L, U, P, Q, vec_rhs);
    std::cout << "RocSolverRf setup completed\n";

    // Test refactorization with the same matrix (in practice, you'd change matrix values)
    std::cout << "\nTesting refactorization with same matrix...\n";

    // Refactorize
    status = Rf.refactorize();
    std::cout << "RocSolverRf refactorization status: " << status << std::endl;
    if (status != 0)
    {
      std::cerr << "Refactorization failed!" << std::endl;
    }
    else
    {
      // Solve with refactorization
      status = Rf.solve(vec_rhs, vec_x);
      std::cout << "RocSolverRf solve status: " << status << std::endl;
      helper.resetSystem(A, vec_rhs, vec_x);

      if (status == 0)
      {
        std::cout << "System solved successfully with refactorization\n";

        // Get solution
        vec_x->syncData(ReSolve::memory::HOST);
        solution = vec_x->getData(ReSolve::memory::HOST);

        std::cout << "Refactorization solution vector:\n";
        std::cout << "[";
        for (int i = 0; i < n; ++i)
        {
          std::cout << solution[i] << (i < n - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Get the rhs
        double* rhs_solution = vec_rhs->getData(ReSolve::memory::HOST);
        std::cout << "Right-hand side vector after solve:\n";
        std::cout << "[";
        for (int i = 0; i < n; ++i)
        {
          std::cout << rhs_solution[i] << (i < n - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
      }
    }
  }

  // Cleanup
  delete A;
  delete vec_x;
  delete vec_rhs;

  std::cout << "Example completed successfully!\n";
  return 0;
}
