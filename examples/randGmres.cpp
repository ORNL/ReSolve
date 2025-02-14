#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeRandFGMRES.hpp>
#include <resolve/LinSolverDirectCpuILU0.hpp>
#include <resolve/LinSolverDirectSerialILU0.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#include "ExampleHelper.hpp"

#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSparseILU0.hpp>
#endif

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSparseILU0.hpp>
#endif

/// Prototype of the example main function 
template <class workspace_type, class precon_type>
static int example(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  int status = 0;

  status += example<ReSolve::LinAlgWorkspaceCpu,
                    ReSolve::LinSolverDirectCpuILU0>(argc, argv);

#ifdef RESOLVE_USE_HIP
  status += example<ReSolve::LinAlgWorkspaceHIP,
                    ReSolve::LinSolverDirectRocSparseILU0>(argc, argv);
#endif

#ifdef RESOLVE_USE_CUDA
  status += example<ReSolve::LinAlgWorkspaceCUDA,
                    ReSolve::LinSolverDirectCuSparseILU0>(argc, argv);
#endif

  return status;
}

/// Example implementation
template <class workspace_type, class precon_type>
int example(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using namespace ReSolve;
  using namespace ReSolve::constants;
  using namespace ReSolve::examples;
  using vector_type = ReSolve::vector::Vector;

  (void) argc; // TODO: Check if the number of input parameters is correct.
  std::string  matrix_filename = argv[1];
  std::string  rhsFileName = argv[2];


  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);

  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);

  precon_type Precond(&workspace);
  LinSolverIterativeRandFGMRES FGMRES(&matrix_handler,
                                      &vector_handler,
                                      LinSolverIterativeRandFGMRES::cs,
                                      &GS);

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

  matrix::Csr* A = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  std::ifstream mat_file(matrix_filename);
  if(!mat_file.is_open())
  {
    std::cout << "Failed to open file " << matrix_filename << "\n";
    return -1;
  }
  std::ifstream rhs_file(rhsFileName);
  if(!rhs_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName << "\n";
    return -1;
  }
  bool is_expand_symmetric = true;
  A = io::createCsrFromFile(mat_file, is_expand_symmetric);
  vec_rhs = io::createVectorFromFile(rhs_file);
  mat_file.close();
  rhs_file.close();

  vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(memspace);
  A->syncData(memspace);
  vec_rhs->syncData(memspace);

  printSystemInfo(matrix_filename, A);

  matrix_handler.setValuesChanged(true, memspace);

  Precond.setup(A);
  FGMRES.setRestart(150);
  FGMRES.setMaxit(2500);
  FGMRES.setTol(1e-12);
  FGMRES.setup(A);
  GS.setup(FGMRES.getKrand(), FGMRES.getRestart()); 

  FGMRES.resetMatrix(A);
  FGMRES.setupPreconditioner("LU", &Precond);
  FGMRES.setFlexible(1); 

  FGMRES.solve(vec_rhs, vec_x);

  // Print summary of results
  helper.resetSystem(A, vec_rhs, vec_x);
  std::cout << "\nRandomized GMRES result on " << hwbackend << "\n";
  std::cout << "---------------------------------\n";
  helper.printIrSummary(&FGMRES);

  delete A;
  delete vec_rhs;
  delete vec_x;

  return 0;
}
