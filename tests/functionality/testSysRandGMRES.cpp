/**
 * @file testSysRandGMRES.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for SystemSolver and randomized GMRES classes 
 * @date 2023-12-18
 * 
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/SystemSolver.hpp>

#if defined (RESOLVE_USE_CUDA)
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceCUDA;
  std::string memory_space("cuda");
#elif defined (RESOLVE_USE_HIP)
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
  using workspace_type = ReSolve::LinAlgWorkspaceHIP;
  std::string memory_space("hip");
#else
  using workspace_type = ReSolve::LinAlgWorkspaceCpu;
  std::string memory_space("cpu");
#endif

using namespace ReSolve::constants;

// Use ReSolve data types.
using real_type  = ReSolve::real_type;
using index_type  = ReSolve::index_type;
using vector_type = ReSolve::vector::Vector;

ReSolve::matrix::Csr* generateMatrix(const index_type N);
ReSolve::vector::Vector* generateRhs(const index_type N);

int main(int argc, char *argv[])
{
  // Error sum needs to be 0 at the end for test to PASS.
  // It is a FAIL otheriwse.
  int error_sum = 0;
  int status = 0;

  // Optionally take the matrix size as the input
  const index_type N = (argc == 2) ? atoi(argv[1]) : 10000;

  // Generate linear system data
  ReSolve::matrix::Csr* A = generateMatrix(N);
  vector_type* vec_rhs = generateRhs(N);

  // Create workspace and initialize its handles.
  workspace_type workspace;
  workspace.initializeHandles();

  // Create linear algebra handlers
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);

  // Create system solver
  ReSolve::SystemSolver* solver = new ReSolve::SystemSolver(&workspace, "none", "none", "randgmres", "ilu0", "none");

  // Create solution vector
  vector_type* vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(ReSolve::memory::HOST);

  // Set the initial guess to 0
  vec_x->allocate(ReSolve::memory::DEVICE);
  vec_x->setToZero(ReSolve::memory::DEVICE);

  // Norm of rhs vector
  real_type norm_b = 0.0;

  // Set solver options
  solver->getIterativeSolver().setRestart(200);
  solver->getIterativeSolver().setMaxit(2500);
  solver->getIterativeSolver().setTol(1e-12);

  matrix_handler.setValuesChanged(true, ReSolve::memory::DEVICE);

  // Set system matrix and initialize iterative solver
  status = solver->setMatrix(A);
  error_sum += status;

  // Set preconditioner (default in this case ILU0)
  status = solver->preconditionerSetup();
  error_sum += status;

  // Solve system
  status = solver->solve(vec_rhs, vec_x);
  error_sum += status;

  // Get residual norm
  norm_b = vector_handler.dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE);
  norm_b = std::sqrt(norm_b);
  real_type final_norm_first =  solver->getIterativeSolver().getFinalResidualNorm();
  std::cout << std::scientific << std::setprecision(16)
            << "Randomized FGMRES results (first run): \n"
            << "\t Sketching method:                                    : "
            << "CountSketch\n" 
            << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
            << solver->getIterativeSolver().getInitResidualNorm() << " \n"
            << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
            << solver->getIterativeSolver().getInitResidualNorm()/norm_b <<  " \n"
            << "\t Final residual norm:            ||b-Ax||_2           : " 
            << solver->getIterativeSolver().getFinalResidualNorm() << " \n"
            << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
            << solver->getIterativeSolver().getFinalResidualNorm()/norm_b << " \n"
            << "\t Number of iterations                                 : " 
            << solver->getIterativeSolver().getNumIter() << "\n";

  delete solver;

  // Create a new solver using sketching based on Walsh-Hadamard transform
  solver = new ReSolve::SystemSolver(&workspace, "none", "none", "randgmres", "ilu0", "none");

  // Set solver options
  solver->setSketchingMethod("fwht");
  solver->getIterativeSolver().setRestart(150);
  solver->getIterativeSolver().setMaxit(2500);
  solver->getIterativeSolver().setTol(1e-12);

  // Set system matrix and initialize iterative solver
  status = solver->setMatrix(A);
  error_sum += status;

  // Set preconditioner (default in this case ILU0)
  status = solver->preconditionerSetup();
  error_sum += status;

  // Set the initial guess to 0
  vec_x->setToZero(ReSolve::memory::DEVICE);

  // Solve system
  status = solver->solve(vec_rhs, vec_x);
  error_sum += status;

  // Get residual norm
  std::cout << std::scientific << std::setprecision(16)
            << "Randomized FGMRES results (second run): \n"
            << "\t Sketching method:                                    : "
            << "FWHT\n" 
            << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
            << solver->getIterativeSolver().getInitResidualNorm() << " \n"
            << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
            << solver->getIterativeSolver().getInitResidualNorm()/norm_b <<  " \n"
            << "\t Final residual norm:            ||b-Ax||_2           : " 
            << solver->getIterativeSolver().getFinalResidualNorm() << " \n"
            << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
            << solver->getIterativeSolver().getFinalResidualNorm()/norm_b << " \n"
            << "\t Number of iterations                                 : " 
            << solver->getIterativeSolver().getNumIter() << "\n";

  if ((final_norm_first/norm_b > 1e-11) || (solver->getIterativeSolver().getFinalResidualNorm()/norm_b > 1e-11 )) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout<<"Test randomized GMRES PASSED"<<std::endl<<std::endl;;
  } else {
    std::cout<<"Test randomized GMRES FAILED, error sum: "<<error_sum<<std::endl<<std::endl;;
  }

  delete A;
  delete vec_x;
  delete vec_rhs;

  return error_sum;
}


ReSolve::vector::Vector* generateRhs(const index_type N)
{
  vector_type* vec_rhs = new vector_type(N);
  vec_rhs->allocate(ReSolve::memory::HOST);
  vec_rhs->allocate(ReSolve::memory::DEVICE);

  real_type* data = vec_rhs->getData(ReSolve::memory::HOST);
  for (int i = 0; i < N; ++i) {
    if (i % 2) {
      data[i] = 1.0;
    } else {

      data[i] = -111.0;
    }
  }
  vec_rhs->copyData(ReSolve::memory::HOST, ReSolve::memory::DEVICE);
  return vec_rhs;
} 

ReSolve::matrix::Csr* generateMatrix(const index_type N)
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
  ReSolve::matrix::Csr* A = new ReSolve::matrix::Csr(N, N, NNZ);
  A->allocateMatrixData(ReSolve::memory::HOST);

  index_type* rowptr = A->getRowData(ReSolve::memory::HOST);
  index_type* colidx = A->getColData(ReSolve::memory::HOST);
  real_type* val     = A->getValues(ReSolve::memory::HOST); 

  // Populate CSR matrix using same row pattern as for NNZ calculation
  rowptr[0] = 0;
  index_type where;
  real_type what;
  for (index_type i=0; i < N; ++i)
  {
    size_t reminder = static_cast<size_t>(i%5);
    const std::vector<real_type>& row_sample = data[reminder];
    index_type nnz_per_row = static_cast<index_type>(row_sample.size());

    rowptr[i+1] = rowptr[i] + nnz_per_row;
    bool c = false;
    for (index_type j = rowptr[i]; j < rowptr[i+1]; ++j)
    {
      if (((!c) && (((j - rowptr[i]) * N/nnz_per_row + (N%(N/nnz_per_row))) >= i)) || ((!c) && (j == (rowptr[i+1] - 1)) )) {
        c = true;
        where = i;
        what = 4.;
      } else {
        where =  (j - rowptr[i]) * N/nnz_per_row + (N%(N/nnz_per_row));
        what = row_sample[static_cast<size_t>(j - rowptr[i])];
      } 
      colidx[j] = where;
      // evenly distribute nonzeros ^^^^             ^^^^^^^^ perturb offset
      val[j] = what;
    }
  }


  A->setUpdated(ReSolve::memory::HOST);
  A->copyData(ReSolve::memory::DEVICE);
  return A;
}
