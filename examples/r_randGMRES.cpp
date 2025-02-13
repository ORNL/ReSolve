#include <string>
#include <iostream>
#include <iomanip>

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


#include "ExampleHelper.hpp"

/// Prototype of the example main function 
template <class workspace_type>
static int example(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  return example<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
}

/// Example implementation
template <class workspace_type>
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

  matrix::Csr* A = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);

  LinSolverDirectRocSparseILU0 Precond(&workspace);
  LinSolverIterativeRandFGMRES FGMRES(&matrix_handler,
                                      &vector_handler,
                                      LinSolverIterativeRandFGMRES::cs,
                                      &GS);

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
  vec_x->allocate(memory::DEVICE);
  A->syncData(memory::DEVICE);
  vec_rhs->syncData(memory::DEVICE);

  printSystemInfo(matrix_filename, A);

  matrix_handler.setValuesChanged(true, memory::DEVICE);

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
  helper.printIrSummary(&FGMRES);

  delete A;
  delete vec_rhs;
  delete vec_x;

  return 0;
}
