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
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  (void) argc; // TODO: Check if the number of input parameters is correct.
  std::string  matrix_filename = argv[1];
  std::string  rhsFileName = argv[2];


  workspace_type* workspace = new workspace_type();
  workspace->initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(*workspace);

  MatrixHandler* matrix_handler =  new MatrixHandler(workspace);
  VectorHandler* vector_handler =  new VectorHandler(workspace);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  matrix::Csr* A = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  // vector_type* vec_r   = nullptr;

  GramSchmidt* GS = new GramSchmidt(vector_handler, GramSchmidt::CGS2);

  LinSolverDirectRocSparseILU0* Rf = new LinSolverDirectRocSparseILU0(workspace);
  LinSolverIterativeRandFGMRES* FGMRES = new LinSolverIterativeRandFGMRES(matrix_handler, vector_handler,LinSolverIterativeRandFGMRES::cs, GS);

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

  rhs = io::createArrayFromFile(rhs_file);
  x = new real_type[A->getNumRows()];
  vec_rhs = new vector_type(A->getNumRows());
  vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(memory::HOST);
  //iinit guess is 0U
  vec_x->allocate(memory::DEVICE);
  vec_x->setToZero(memory::DEVICE);
  // vec_r = new vector_type(A->getNumRows());

  mat_file.close();
  rhs_file.close();

  A->syncData(memory::DEVICE);
  vec_rhs->copyDataFrom(rhs, memory::HOST, memory::DEVICE);

  printSystemInfo(matrix_filename, A);

  //Now call the solver
  // real_type norm_b;
  matrix_handler->setValuesChanged(true, memory::DEVICE);

  Rf->setup(A);
  FGMRES->setRestart(150);
  FGMRES->setMaxit(2500);
  FGMRES->setTol(1e-12);
  FGMRES->setup(A);
  GS->setup(FGMRES->getKrand(), FGMRES->getRestart()); 

  //matrix_handler->setValuesChanged(true, memory::DEVICE);
  FGMRES->resetMatrix(A);
  FGMRES->setupPreconditioner("LU", Rf);
  FGMRES->setFlexible(1); 

  vec_rhs->copyDataFrom(rhs, memory::HOST, memory::DEVICE);
  FGMRES->solve(vec_rhs, vec_x);

  // Print summary of results
  helper.resetSystem(A, vec_rhs, vec_x);
  helper.printIrSummary(FGMRES);

  delete A;
  delete Rf;
  delete [] x;
  delete [] rhs;
  delete vec_rhs;
  delete vec_x;
  delete workspace;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
