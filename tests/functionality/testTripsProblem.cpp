/**
 * @file testTripsProblem.cpp
 * @author Kasia Swirydowicz (kasia.swirydowicz@pnnl.gov)
 * @author Slaven Peles (peless@ornl.gov)
 * @author Jeffery Zhang (jeffzsc594@gmail.com)
 * @brief Functionality test for GMRES using test problem from Trips-PY
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
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/PreconditionerLU.hpp>

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
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Create test linear system using given arguments
  if (argc != 4) {
    std::cout << "\nMust provide the following files: A matrix file, rhs file\n"
              << "A^T file\n";
  }

  std::ifstream A_file(argv[1]);
  std::ifstream rhs_file(argv[2]);
  std::ifstream At_file(argv[3]);

  matrix::Csr* A = ReSolve::io::createCsrFromFile(A_file, false);
  vector_type* vec_rhs = ReSolve::io::createVectorFromFile(rhs_file);

  // Declare A_t
  matrix::Csr* A_t = ReSolve::io::createCsrFromFile(At_file, false);

  vector_type vec_x(A->getNumRows());
  vec_x.allocate(memory::HOST);
  vec_x.allocate(memspace);
  vec_x.setToZero(memspace);

  // Send A, A_t and b to gpu memspace
  if (matrix_handler.getIsCudaEnabled()) {
    A->allocateMatrixData(memspace);
    A_t->allocateMatrixData(memspace);
    vec_rhs->allocate(memspace);

    A->syncData(memspace);
    A_t->syncData(memspace);
    vec_rhs->syncData(memspace);
  }

  // Obtain A_t for AB and BA GMRES
  error_sum += matrix_handler.transpose(A, A_t, memspace);

  matrix_handler.setValuesChanged(true, memspace);

  real_type tol = 1e-10; // iterative solver tolerance

  // Compute error norms for the system
  helper.setSystem(A, vec_rhs, &vec_x);

  matrix_handler.setValuesChanged(true, memspace);

  // Configure preconditioner
  status = ILU.setup(A);
  error_sum += status;

  // Set solver parameters
  FGMRES.setMaxit(2500);
  FGMRES.setTol(tol);
  FGMRES.setup(A);

  // Use A_t as the preconditioner
  status = FGMRES.setPreconditioner(A_t);
  error_sum += status;

  // Use standard GMRES
  FGMRES.setFlexible(false);
  
  std::cout << FGMRES.getPreconditionerDir() << "\n\n";
  std::cout << FGMRES.getPreconditionerType() << "\n\n";
  std::cout << FGMRES.getFlexible() << "\n\n";

  // Default uses ABGMRES
  FGMRES.setRestart(150);
  // 6 Iterations is used to obtain a good solution despite semi-convergence
  FGMRES.setMaxit(6);
  FGMRES.setTol(tol);

  helper.resetSystem(A, vec_rhs, &vec_x);
  vec_x.setToZero(memspace);
  status = FGMRES.solve(vec_rhs, &vec_x);
  error_sum += status;

  // Print result summary and check solution
  std::cout << "\n  ABGMRES results: \n"
            << "\t Hardware backend:                              : "
            << hwbackend << "\n";
  helper.printIterativeSolverSummary(&FGMRES);
  std::ofstream AB_output_file("AB_output.mtx");
  error_sum += ReSolve::io::writeVectorToFile(&vec_x, AB_output_file);

  // Change preconditioner direction for BAGMRES
  status = FGMRES.setPreconditionerDir("left");
  error_sum += status;

  // Use BAGMRES
  FGMRES.setRestart(150);
  FGMRES.setMaxit(6);
  FGMRES.setTol(tol);

  helper.resetSystem(A, vec_rhs, &vec_x);
  vec_x.setToZero(memspace);
  status = FGMRES.solve(vec_rhs, &vec_x);
  error_sum += status;

  // Print result summary and check solution
  std::cout << "\n  BAGMRES results: \n"
            << "\t Hardware backend:                              : "
            << hwbackend << "\n";
  helper.printIterativeSolverSummary(&FGMRES);
  std::ofstream BA_output_file("BA_output.mtx");
  error_sum += ReSolve::io::writeVectorToFile(&vec_x, BA_output_file);

  isTestPass(error_sum, "Test ABBAGMRES on " + hwbackend + " device");

  delete A;
  delete A_t;
  delete vec_rhs;

  return error_sum;
}

