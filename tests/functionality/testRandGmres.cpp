/**
 * @file testRandGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for randomized GMRES class with CUDA backend.
 *
 */
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#endif

using real_type   = ReSolve::real_type;
using index_type  = ReSolve::index_type;
using vector_type = ReSolve::vector::Vector;
using MemorySpace = ReSolve::memory::MemorySpace;

#include "TestHelper.hpp"

static ReSolve::matrix::Csr*    generateMatrix(const index_type n, MemorySpace memspace);
static ReSolve::vector::Vector* generateRhs(const index_type n, MemorySpace memspace);

template <class workspace_type, class preconditioner_type>
static int runTest(int argc, char* argv[]);

int main(int argc, char* argv[])
{
  int error_sum = 0; // If error sum is 0, test passes; fails otherwise

  error_sum += runTest<ReSolve::LinAlgWorkspaceCpu,
                       ReSolve::LinSolverDirectCpuILU0>(argc, argv);

#ifdef RESOLVE_USE_CUDA
  error_sum += runTest<ReSolve::LinAlgWorkspaceCUDA,
                       ReSolve::LinSolverDirectCuSparseILU0>(argc, argv);
#endif

#ifdef RESOLVE_USE_HIP
  error_sum += runTest<ReSolve::LinAlgWorkspaceHIP,
                       ReSolve::LinSolverDirectRocSparseILU0>(argc, argv);
#endif

  return error_sum;
}

template <class workspace_type, class preconditioner_type>
int runTest(int argc, char* argv[])
{
  using namespace ReSolve;
  int error_sum = 0; // If error sum is 0, test passes; fails otherwise
  int status;

  workspace_type workspace;
  workspace.initializeHandles();

  // Create test helper
  TestHelper<workspace_type> helper(workspace);

  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  // Set memory space where to run tests
  std::string         hwbackend = "CPU";
  memory::MemorySpace memspace  = memory::HOST;
  if (matrix_handler.getIsCudaEnabled())
  {
    memspace  = memory::DEVICE;
    hwbackend = "CUDA";
  }
  if (matrix_handler.getIsHipEnabled())
  {
    memspace  = memory::DEVICE;
    hwbackend = "HIP";
  }

  // Create iterative solver
  GramSchmidt                                   GS(&vector_handler, GramSchmidt::CGS2);
  preconditioner_type                           ILU(&workspace);
  LinSolverIterativeRandFGMRES::SketchingMethod sketching =
      LinSolverIterativeRandFGMRES::cs;
  LinSolverIterativeRandFGMRES FGMRES(&matrix_handler,
                                      &vector_handler,
                                      sketching,
                                      &GS);

  // Create test linear system (default size 10,000)
  const index_type n       = (argc == 2) ? atoi(argv[1]) : 10000;
  matrix::Csr*     A       = generateMatrix(n, memspace);
  vector_type*     vec_rhs = generateRhs(n, memspace);

  vector_type vec_x(A->getNumRows());
  vec_x.allocate(memspace);
  vec_x.setToZero(memspace);

  matrix_handler.setValuesChanged(true, memspace);

  real_type tol           = 1e-12;      // iterative solver tolerance
  real_type test_pass_tol = 10.0 * tol; // test results tolerance

  // Configure preconditioner
  status = ILU.setup(A);
  error_sum += status;

  // Set solver parameters
  FGMRES.setMaxit(2500);
  FGMRES.setTol(tol);
  FGMRES.setup(A);

  // Typically, you would want these settings _before_ matrix A setup, but here we test
  // flexibility of Re::Solve configuration options
  FGMRES.setRestart(200);
  FGMRES.setSketchingMethod(LinSolverIterativeRandFGMRES::cs);

  status = FGMRES.setupPreconditioner("LU", &ILU);
  error_sum += status;

  FGMRES.setFlexible(true);

  status = FGMRES.solve(vec_rhs, &vec_x);
  error_sum += status;

  // Compute error norms for the system
  helper.setSystem(A, vec_rhs, &vec_x);

  // Print result summary and check solution
  std::cout << "\nRandomized FGMRES results: \n"
            << "\t Hardware backend:                              : "
            << hwbackend << "\n"
            << "\t Sketching method:                              : "
            << "CountSketch\n";
  helper.printIterativeSolverSummary(&FGMRES);
  error_sum += helper.checkResult(test_pass_tol);

  // Change sketching method for the existing randomized GMRES solver
  FGMRES.setSketchingMethod(LinSolverIterativeRandFGMRES::fwht);
  FGMRES.setRestart(150);
  FGMRES.setMaxit(2500);
  FGMRES.setTol(tol);
  FGMRES.resetMatrix(A);

  vec_x.setToZero(memspace);
  status = FGMRES.solve(vec_rhs, &vec_x);
  error_sum += status;

  // Print result summary and check solution
  std::cout << "\nRandomized FGMRES results: \n"
            << "\t Hardware backend:                              : "
            << hwbackend << "\n"
            << "\t Sketching method:                              : "
            << "FWHT\n";
  helper.printIterativeSolverSummary(&FGMRES);
  error_sum += helper.checkResult(test_pass_tol);

  isTestPass(error_sum, "Test Randomized GMRES on " + hwbackend + " device");

  delete A;
  delete vec_rhs;

  return error_sum;
}

ReSolve::vector::Vector* generateRhs(const index_type             n,
                                     ReSolve::memory::MemorySpace memspace)
{
  vector_type* vec_rhs = new vector_type(n);
  vec_rhs->allocate(ReSolve::memory::HOST);

  real_type* data = vec_rhs->getData(ReSolve::memory::HOST);
  for (int i = 0; i < n; ++i)
  {
    if (i % 2)
    {
      data[i] = 1.0;
    }
    else
    {

      data[i] = -111.0;
    }
  }
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  if (memspace != ReSolve::memory::HOST)
  {
    vec_rhs->syncData(memspace);
  }
  return vec_rhs;
}

/**
 * @brief Generates a sparse matrix in CSR format with cyclic row patterns
 *
 * Creates an n×n sparse matrix with nonzero values following a cyclic pattern
 * across 5 predefined row templates. Each row is guaranteed to have a diagonal
 * entry with value 4.0. Off-diagonal values are distributed approximately evenly
 * across columns.
 *
 * @param n Dimension of the square matrix (n×n)
 * @param memory_space Target memory location (HOST, DEVICE, etc.)
 * @return Pointer to newly allocated CSR matrix
 */
ReSolve::matrix::Csr* generateMatrix(const index_type n, ReSolve::memory::MemorySpace memory_space)
{
  // Define 5 row patterns that cycle through the matrix
  // Each pattern's values sum to 30 for consistent row sums
  const std::vector<std::vector<real_type>> row_patterns = {
      {1., 5., 7., 8., 3., 2., 4.},                 // 7 nonzeros, sum = 30
      {1., 3., 2., 2., 1., 6., 7., 3., 2., 3.},     // 10 nonzeros, sum = 30
      {11., 15., 4.},                               // 3 nonzeros, sum = 30
      {1., 1., 5., 1., 9., 2., 1., 2., 3., 2., 3.}, // 11 nonzeros, sum = 30
      {6., 5., 7., 3., 2., 5., 2.}                  // 7 nonzeros, sum = 30
  };

  size_t full_cycles    = static_cast<size_t>(n) / 5;
  size_t remaining_rows = static_cast<size_t>(n) % 5;

  // Calculate total number of nonzeros by cycling through patterns
  size_t total_nonzeros = full_cycles * 38; // 38 nonzeros per full cycle of 5 rows
  for (size_t i = 0; i < remaining_rows; ++i)
  {
    total_nonzeros += static_cast<size_t>(row_patterns[i].size());
  }

  // Allocate CSR matrix structure
  ReSolve::matrix::Csr* matrix = new ReSolve::matrix::Csr(n, n, static_cast<index_type>(total_nonzeros));
  matrix->allocateMatrixData(ReSolve::memory::HOST);

  // Get pointers to CSR data structures
  index_type* row_offsets    = matrix->getRowData(ReSolve::memory::HOST);
  index_type* column_indices = matrix->getColData(ReSolve::memory::HOST);
  real_type*  values         = matrix->getValues(ReSolve::memory::HOST);

  // Populate CSR matrix row by row
  row_offsets[0] = 0;
  bool diagonal_placed;

  for (index_type row = 0; row < n; ++row)
  {
    size_t                        pattern_index   = static_cast<size_t>(row % 5);
    const std::vector<real_type>& current_pattern = row_patterns[pattern_index];
    index_type                    nonzeros_in_row = static_cast<index_type>(current_pattern.size());

    row_offsets[row + 1] = row_offsets[row] + nonzeros_in_row;
    diagonal_placed      = false;

    // Place nonzeros for this row
    for (index_type nz_index = row_offsets[row]; nz_index < row_offsets[row + 1]; ++nz_index)
    {
      index_type position_in_row = nz_index - row_offsets[row];
      index_type column;
      real_type  value;

      // Determine if this position should contain the diagonal element
      index_type estimated_column = position_in_row * n / nonzeros_in_row
                                    + (n % (n / nonzeros_in_row));
      bool is_last_nonzero      = (nz_index == row_offsets[row + 1] - 1);
      bool column_past_diagonal = (estimated_column >= row);

      if (!diagonal_placed && (column_past_diagonal || is_last_nonzero))
      {
        diagonal_placed = true;
        column          = row;
        value           = 4.0;
      }
      else
      {
        // Place off-diagonal element, distributing columns evenly
        column = estimated_column;
        value  = current_pattern[static_cast<size_t>(position_in_row)];
      }

      column_indices[nz_index] = column;
      values[nz_index]         = value;
    }
  }

  // Mark data as updated and sync to target memory space if needed
  matrix->setUpdated(ReSolve::memory::HOST);
  if (memory_space != ReSolve::memory::HOST)
  {
    matrix->syncData(memory_space);
  }

  return matrix;
}
