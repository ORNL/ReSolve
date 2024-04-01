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
#include <cmath>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/utilities/params/CliOptions.hpp>

using namespace ReSolve::constants;
using namespace ReSolve::colors;

// Use ReSolve data types.
using real_type  = ReSolve::real_type;
using index_type  = ReSolve::index_type;
using vector_type = ReSolve::vector::Vector;

// Forward declarations of helper functions that create test linear system
template <class T>
static int test(int argc, char *argv[]);
static void processInputs(std::string& method, std::string& gs, std::string& sketch);
static std::string headerInfo(const std::string& method, const std::string& gs, const std::string& sketch, bool flexible);
static ReSolve::matrix::Csr* generateMatrix(const index_type N, ReSolve::memory::MemorySpace memspace);
static ReSolve::vector::Vector* generateRhs(const index_type N, ReSolve::memory::MemorySpace memspace);

int main(int argc, char *argv[])
{
  int error_sum = 0;

  error_sum += test<ReSolve::LinAlgWorkspaceCpu>(argc, argv);

#ifdef RESOLVE_USE_HIP
  error_sum += test<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
#endif

#ifdef RESOLVE_USE_CUDA
  error_sum += test<ReSolve::LinAlgWorkspaceCUDA>(argc, argv);
#endif

  return error_sum;
}

template <class T>
int test(int argc, char *argv[])
{
  // Error sum needs to be 0 at the end for test to PASS.
  // It is a FAIL otheriwse.
  int error_sum = 0;
  int status = 0;

  // Collect all CLI
  ReSolve::CliOptions options(argc, argv);
  ReSolve::CliOptions::Option* opt = nullptr;

  opt = options.getParamFromKey("-N");
  const index_type N = opt ? atoi((*opt).second.c_str()) : 10000;

  opt = options.getParamFromKey("-i");
  std::string method = opt ? (*opt).second : "randgmres";

  opt = options.getParamFromKey("-g");
  std::string gs = opt ? (*opt).second : "cgs2";

  opt = options.getParamFromKey("-s");
  std::string sketch = opt ? (*opt).second : "count";

  opt = options.getParamFromKey("-x");
  bool flexible = true;
  if(opt) {
    flexible = !((*opt).second == "no");
  }

  processInputs(method, gs, sketch);

  // Create workspace and initialize its handles.
  T workspace;
  workspace.initializeHandles();

  // Create linear algebra handlers
  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);

  // Set memory space where to run tests
  std::string hwbackend = "CPU";
  ReSolve::memory::MemorySpace memspace = ReSolve::memory::HOST;
  if (matrix_handler.getIsCudaEnabled()) {
    memspace = ReSolve::memory::DEVICE;
    hwbackend = "CUDA";
  }
  if (matrix_handler.getIsHipEnabled()) {
    memspace = ReSolve::memory::DEVICE;
    hwbackend = "HIP";
  }

  // Create system solver
  ReSolve::SystemSolver solver(&workspace, "none", "none", method, "ilu0", "none");
  solver.setGramSchmidtMethod(gs);

  // Generate linear system data
  ReSolve::matrix::Csr* A = generateMatrix(N, memspace);
  vector_type* vec_rhs = generateRhs(N, memspace);

  // Create solution vector
  vector_type vec_x(A->getNumRows());
  vec_x.allocate(ReSolve::memory::HOST);

  // Set the initial guess to 0
  vec_x.allocate(memspace);
  vec_x.setToZero(memspace);

  // Norm of rhs vector
  real_type norm_b = 0.0;

  // Set solver options
  real_type tol = 1e-12;
  solver.getIterativeSolver().setMaxit(2500);
  solver.getIterativeSolver().setTol(tol);
  solver.getIterativeSolver().setRestart(200);

  matrix_handler.setValuesChanged(true, memspace);

  // Set system matrix and initialize iterative solver
  status = solver.setMatrix(A);
  error_sum += status;

  if (method == "randgmres") {
    solver.setSketchingMethod(sketch);
  }
  solver.getIterativeSolver().setFlexible(flexible);

  // Set preconditioner (default in this case ILU0)
  status = solver.preconditionerSetup();
  error_sum += status;

  // Solve system
  status = solver.solve(vec_rhs, &vec_x);
  error_sum += status;

  // Get residual norm
  norm_b = vector_handler.dot(vec_rhs, vec_rhs, memspace);
  norm_b = std::sqrt(norm_b);
  real_type final_norm = solver.getIterativeSolver().getFinalResidualNorm();
  std::cout << std::defaultfloat
            << headerInfo(method, gs, sketch, flexible)
            << "\t Hardware backend:               " << hwbackend << "\n"
            << "\t Solver tolerance:               " << tol       << "\n"
            << std::scientific << std::setprecision(16)
            << "\t Initial residual norm:          ||b-Ax_0||_2         : " 
            << solver.getIterativeSolver().getInitResidualNorm() << " \n"
            << "\t Initial relative residual norm: ||b-Ax_0||_2/||b||_2 : "
            << solver.getIterativeSolver().getInitResidualNorm()/norm_b <<  " \n"
            << "\t Final residual norm:            ||b-Ax||_2           : " 
            << solver.getIterativeSolver().getFinalResidualNorm() << " \n"
            << "\t Final relative residual norm:   ||b-Ax||_2/||b||_2   : " 
            << solver.getIterativeSolver().getFinalResidualNorm()/norm_b << " \n"
            << "\t Number of iterations                                 : " 
            << solver.getIterativeSolver().getNumIter() << "\n";

  if (final_norm/norm_b > (10.0 * tol)) {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0) {
    std::cout << "Test " << GREEN << "PASSED" << CLEAR << "\n\n";
  } else {
    std::cout << "Test " << RED << "FAILED" << CLEAR << ", error sum: " << error_sum << "\n\n";
  }

  delete A;
  delete vec_rhs;

  return error_sum;
}

void processInputs(std::string& method, std::string& gs, std::string& sketch)
{
  if (method == "randgmres") {
    if ((sketch != "count") && (sketch != "fwht")) {
      std::cout << "Sketching method " << sketch << " not recognized.\n";
      std::cout << "Setting sketch to the default (count).\n\n";
      sketch = "count";
    }
  }

  if ((method != "randgmres") && (method != "fgmres")) {
    std::cout << "Unknown method " << method << "\n";
    std::cout << "Setting iterative solver method to the default (FGMRES).\n\n";
    method = "fgmres";
  }

  if (gs != "cgs1" && gs != "cgs2" && gs != "mgs" && gs != "mgs_two_synch" && gs != "mgs_pm") {
    std::cout << "Unknown orthogonalization " << gs << "\n";
    std::cout << "Setting orthogonalization to the default (CGS2).\n\n";
    gs = "cgs2";
  }
}

std::string headerInfo(const std::string& method, const std::string& gs, const std::string& sketch, bool flexible)
{
  std::string header("Results for ");
  if (method == "randgmres") {
    header += "randomized ";
    header += flexible ? "FGMRES" : "GMRES";
    header += " solver\n";
    header += "\t Sketching method:               ";
    if (sketch == "count") {
      header += "count sketching\n";
    } else if (sketch == "fwht") {
      header += "fast Walsh-Hadamard transform\n";
    }
  } else if (method == "fgmres") {
    header += flexible ? "FGMRES" : "GMRES";
    header += " solver\n";
  } else {
    return header + "unknown method\n";
  }

  std::string withgs = "\t Orthogonalization method:       ";
  if (gs == "cgs2") {
    header += (withgs + "reorthogonalized classical Gram-Schmidt\n");
  } else if (gs == "cgs1")  {
    header += (withgs + "classical Gram-Schmidt\n");
  } else if (gs == "mgs") {
    header += (withgs + "modified Gram-Schmidt\n");    
  } else if (gs == "mgs_two_synch") {
    header += (withgs + "modified Gram-Schmidt 2-sync\n");    
  } else if (gs == "mgs_pm") {
    header += (withgs + "post-modern modified Gram-Schmidt\n");    
  } else if (gs == "mgs") {
    header += (withgs + "modified Gram-Schmidt\n");    
  } else {
    // do nothing
  }

  return header;
}

ReSolve::vector::Vector* generateRhs(const index_type N, ReSolve::memory::MemorySpace memspace)
{
  vector_type* vec_rhs = new vector_type(N);
  vec_rhs->allocate(ReSolve::memory::HOST);
  vec_rhs->allocate(memspace);

  real_type* data = vec_rhs->getData(ReSolve::memory::HOST);
  for (int i = 0; i < N; ++i) {
    if (i % 2) {
      data[i] = 1.0;
    } else {

      data[i] = -111.0;
    }
  }
  vec_rhs->copyData(ReSolve::memory::HOST, memspace);
  return vec_rhs;
} 

ReSolve::matrix::Csr* generateMatrix(const index_type N, ReSolve::memory::MemorySpace memspace)
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
  A->copyData(memspace);
  return A;
}
