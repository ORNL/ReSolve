/**
 * @file testRandGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for randomized GMRES class with CUDA backend. 
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#endif

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#endif

using real_type  = ReSolve::real_type;
using index_type  = ReSolve::index_type;
using vector_type = ReSolve::vector::Vector;
using MemorySpace = ReSolve::memory::MemorySpace;

#include "TestHelper.hpp"

static ReSolve::matrix::Csr* generateMatrix(const index_type n, MemorySpace memspace);
static ReSolve::vector::Vector* generateRhs(const index_type n, MemorySpace memspace);

template <class workspace_type, class preconditioner_type>
static int runTest(int argc, char *argv[]);

int main(int argc, char *argv[])
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
int runTest(int argc, char *argv[])
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
  std::string hwbackend = "CPU";
  memory::MemorySpace memspace = memory::HOST;
  if (matrix_handler.getIsCudaEnabled()) {
    memspace = memory::DEVICE;
    hwbackend = "CUDA";
  }
  if (matrix_handler.getIsHipEnabled()) {
    memspace = memory::DEVICE;
    hwbackend = "HIP";
  }

  // Create iterative solver
  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);
  preconditioner_type ILU(&workspace);
  LinSolverIterativeRandFGMRES::SketchingMethod sketching =
    LinSolverIterativeRandFGMRES::cs;
  LinSolverIterativeRandFGMRES FGMRES(&matrix_handler,
                                               &vector_handler,
                                               sketching,
                                               &GS);

  // Create test linear system (default size 10,000)
  const index_type n = (argc == 2) ? atoi(argv[1]) : 10000;
  matrix::Csr* A = generateMatrix(n, memspace);
  vector_type* vec_rhs = generateRhs(n, memspace);

  vector_type vec_x(A->getNumRows());
  vec_x.allocate(memspace);
  vec_x.setToZero(memspace);

  matrix_handler.setValuesChanged(true, memspace);

  real_type tol = 1e-12; // iterative solver tolerance
  real_type test_pass_tol = 10.0*tol; // test results tolerance

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


ReSolve::vector::Vector* generateRhs(const index_type n, 
                                     ReSolve::memory::MemorySpace memspace)
{
  vector_type* vec_rhs = new vector_type(n);
  vec_rhs->allocate(ReSolve::memory::HOST);

  real_type* data = vec_rhs->getData(ReSolve::memory::HOST);
  for (int i = 0; i < n; ++i) {
    if (i % 2) {
      data[i] = 1.0;
    } else {

      data[i] = -111.0;
    }
  }
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  if (memspace != ReSolve::memory::HOST) {
    vec_rhs->syncData(memspace);
  }
  return vec_rhs;
} 

ReSolve::matrix::Csr* generateMatrix(const index_type n, 
                                     ReSolve::memory::MemorySpace memspace)
{
  std::vector<real_type> r1 = {1., 5., 7., 8., 3., 2., 4.}; // sum 30
  std::vector<real_type> r2 = {1., 3., 2., 2., 1., 6., 7., 3., 2., 3.}; // sum 30
  std::vector<real_type> r3 = {11., 15., 4.}; // sum 30
  std::vector<real_type> r4 = {1., 1., 5., 1., 9., 2., 1., 2., 3., 2., 3.}; // sum 30
  std::vector<real_type> r5 = {6., 5., 7., 3., 2., 5., 2.}; // sum 30


  const std::vector<std::vector<real_type> > data = {r1, r2, r3, r4, r5};

  // First compute number of nonzeros
  index_type nnz = 0;
  for (index_type i = 0; i < n; ++i)
  {
    size_t reminder = static_cast<size_t>(i%5);
    nnz += static_cast<index_type>(data[reminder].size());
  }

  // Allocate NxN CSR matrix with nnz nonzeros
  ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(n, n, nnz);
  A->allocateMatrixData(ReSolve::memory::HOST);

  index_type* rowptr = A->getRowData(ReSolve::memory::HOST);
  index_type* colidx = A->getColData(ReSolve::memory::HOST);
  real_type* val     = A->getValues(ReSolve::memory::HOST); 

  // Populate CSR matrix using same row pattern as for nnz calculation
  rowptr[0] = 0;
  index_type where;
  real_type what;
  for (index_type i=0; i < n; ++i)
  {
    size_t reminder = static_cast<size_t>(i%5);
    const std::vector<real_type>& row_sample = data[reminder];
    index_type nnz_per_row = static_cast<index_type>(row_sample.size());

    rowptr[i+1] = rowptr[i] + nnz_per_row;
    bool c = false;
    for (index_type j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      if (((!c) && (((j - rowptr[i]) * n/nnz_per_row + (n%(n/nnz_per_row))) >= i)) || ((!c) && (j == (rowptr[i+1] - 1)) )) {
        c = true;
        where = i;
        what = 4.;
      } else {
        where =  (j - rowptr[i]) * n/nnz_per_row + (n%(n/nnz_per_row));
        // evenly distribute nonzeros ^^^^             ^^^^^^^^ perturb offset
        what = row_sample[static_cast<size_t>(j - rowptr[i])];
      } 
      colidx[j] = where;
      val[j] = what;
    }
  }


  A->setUpdated(ReSolve::memory::HOST);
  if (memspace != ReSolve::memory::HOST) {
    A->syncData(memspace);
  }
  return A;
}
