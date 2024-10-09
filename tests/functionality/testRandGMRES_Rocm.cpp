/**
 * @file testRandGMRES_Rocm.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Functionality test for randomized GMRES class with HIP backend. 
 * 
 */
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using namespace ReSolve::colors;
using real_type  = ReSolve::real_type;
using index_type  = ReSolve::index_type;
using vector_type = ReSolve::vector::Vector;

static ReSolve::matrix::Csr* generateMatrix(const index_type N);
static ReSolve::vector::Vector* generateRhs(const index_type N);

int main(int argc, char *argv[])
{
  int error_sum = 0; // If error sum is 0, test passes; fails otherwise
  int status;

  const index_type N = (argc == 2) ? atoi(argv[1]) : 10000;
  ReSolve::matrix::Csr* A = generateMatrix(N);
  vector_type* vec_rhs = generateRhs(N);

  ReSolve::LinAlgWorkspaceHIP* workspace_HIP = new ReSolve::LinAlgWorkspaceHIP();
  workspace_HIP->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_HIP);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_HIP);

  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);

  auto* Rf = new ReSolve::LinSolverDirectRocSparseILU0(workspace_HIP);
  auto* FGMRES = new ReSolve::LinSolverIterativeRandFGMRES(matrix_handler, vector_handler,ReSolve::LinSolverIterativeRandFGMRES::cs, GS);

  vector_type* vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(ReSolve::memory::HOST);
  vec_x->allocate(ReSolve::memory::DEVICE);
  vec_x->setToZero(ReSolve::memory::DEVICE);

  real_type norm_b;
  matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

  real_type tol = 1e-12;

  status = Rf->setup(A);
  error_sum += status;

  FGMRES->setMaxit(2500);
  FGMRES->setTol(tol);
  FGMRES->setup(A);

  // Typically, you would want these settings _before_ matrix A setup, but here we test
  // flexibility of Re::Solve configuration options
  FGMRES->setRestart(200);
  FGMRES->setSketchingMethod(ReSolve::LinSolverIterativeRandFGMRES::cs);

  error_sum += status;

  status = FGMRES->setupPreconditioner("LU", Rf);
  error_sum += status;

  FGMRES->setFlexible(true); 

  status = FGMRES->solve(vec_rhs, vec_x);
  error_sum += status;

  norm_b = vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE);
  norm_b = std::sqrt(norm_b);
  real_type final_norm_first = FGMRES->getFinalResidualNorm();
  std::cout << "Randomized FGMRES results (first run): \n"
            << "\t Sketching method:                                    : CountSketch\n" 
            << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
            << std::scientific << std::setprecision(16) 
            << FGMRES->getInitResidualNorm()<<" \n"
            << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
            << FGMRES->getInitResidualNorm()/norm_b<<" \n"
            << "\t Final residual norm:            ||b-Ax||_2           : " 
            << FGMRES->getFinalResidualNorm() <<" \n"
            << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
            << FGMRES->getFinalResidualNorm()/norm_b <<" \n"
            << "\t Number of iterations                                 : "
            << FGMRES->getNumIter() << "\n";


  // Change sketching method for the existing randomized GMRES solver
  FGMRES->setSketchingMethod(ReSolve::LinSolverIterativeRandFGMRES::fwht);
  FGMRES->setRestart(150);
  FGMRES->setMaxit(2500);
  FGMRES->setTol(tol);
  FGMRES->resetMatrix(A);

  vec_x->setToZero(ReSolve::memory::DEVICE);
  status = FGMRES->solve(vec_rhs, vec_x);
  error_sum += status;

  real_type final_norm_second = FGMRES->getFinalResidualNorm();
  std::cout << "Randomized FGMRES results (second run): \n"
            << "\t Sketching method:                                    : FWHT\n" 
            << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
            << std::scientific << std::setprecision(16) 
            << FGMRES->getInitResidualNorm()<<" \n"
            << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
            << FGMRES->getInitResidualNorm()/norm_b<<" \n"
            << "\t Final residual norm:            ||b-Ax||_2           : " 
            << FGMRES->getFinalResidualNorm() <<" \n"
            << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
            << FGMRES->getFinalResidualNorm()/norm_b <<" \n"
            << "\t Number of iterations                                 : " << FGMRES->getNumIter() << "\n";

  if (!isfinite(final_norm_first) || !isfinite(final_norm_second)) {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  if ((final_norm_first/norm_b > 10.*tol) || (final_norm_second/norm_b > 10.*tol)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout << "Test randomized GMRES " << GREEN << "PASSED" << CLEAR << "\n\n";
  } else {
    std::cout << "Test randomized GMRES " << RED << "FAILED" << CLEAR << ", error sum: " << error_sum << "\n\n";
  }

  delete A;
  delete Rf;
  delete vec_x;
  delete vec_rhs;
  delete FGMRES;
  delete GS;
  delete matrix_handler;
  delete vector_handler;
  delete workspace_HIP;

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
  vec_rhs->syncData(ReSolve::memory::HOST, ReSolve::memory::DEVICE);
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
  A->syncData(ReSolve::memory::DEVICE);
  return A;
}
