#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include "ExampleHelper.hpp"
using namespace ReSolve::constants;
/// Prototype of the example function
template <class workspace_type, class solve_type>
static int kluStandAlone(int argc, char *argv[]);

/// Main function selects example to be run.
int main(int argc, char *argv[])
{
  kluStandAlone<ReSolve::LinAlgWorkspaceCpu,
             ReSolve::LinSolverDirectKLU>(argc, argv);
  #ifdef RESOLVE_USE_CUDA
    kluStandAlone<ReSolve::LinAlgWorkspaceCUDA,
               ReSolve::LinSolverDirectKLU>(argc, argv);
  #endif

  #ifdef RESOLVE_USE_HIP
    kluStandAlone<ReSolve::LinAlgWorkspaceHIP,
                ReSolve::LinSolverDirectKLU>(argc, argv);
  #endif

  return 0;
}


/**
 * @brief Example of using a stand alone solve on for KLU
 *
 * @tparam workspace_type - Type of the workspace to use
 * @param[in] argc - Number of command line arguments
 * @param[in] argv - Command line arguments
 * @return 0 if the example ran successfully, -1 otherwise
 */
template <class workspace_type, class solve_type>
int kluStandAlone(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  (void) argc; // TODO: Check if the number of input parameters is correct.
  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  std::cout<<"Family mtx file name: "<< matrixFileName << std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << std::endl;

  std::string fileId;
  std::string rhsId;

  ReSolve::matrix::Csr* A = nullptr;
  workspace_type workspace;
  workspace.initializeHandles();
  // Create a helper object (computing errors, printing summaries, etc.)
  ReSolve::examples::ExampleHelper<workspace_type> helper(workspace);
  std::cout << "sysRefactor with " << helper.getHardwareBackend() << " backend\n";

  ReSolve::MatrixHandler matrix_handler(&workspace);
  ReSolve::VectorHandler vector_handler(&workspace);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  vector_type* vec_r   = nullptr;
  real_type norm_A, norm_x, norm_r;//used for INF norm

  solve_type* KLU = new solve_type;



  // Read matrix first
  std::cout << "========================================================================================================================"<<std::endl;
  std::cout << "Reading: " << matrixFileName << std::endl;
  std::cout << "========================================================================================================================"<<std::endl;
  std::cout << std::endl;
  // Read first matrix
  std::ifstream mat_file(matrixFileName);
  if(!mat_file.is_open())
  {
    std::cout << "Failed to open file " << matrixFileName << "\n";
    return -1;
  }
  std::ifstream rhs_file(rhsFileName);
  if(!rhs_file.is_open())
  {
    std::cout << "Failed to open file " << rhsFileName << "\n";
    return -1;
  }
  bool is_expand_symmetric = true;
  A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

  rhs = ReSolve::io::createArrayFromFile(rhs_file);
  x = new real_type[A->getNumRows()];
  vec_rhs = new vector_type(A->getNumRows());
  vec_x = new vector_type(A->getNumRows());
  vec_r = new vector_type(A->getNumRows());
  std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;
  mat_file.close();
  rhs_file.close();

  vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  std::cout << "COO to CSR completed. Expanded NNZ: " << A->getNnz() << std::endl;
  //Now call direct solver
  int status;
  KLU->setup(A);
  status = KLU->analyze();
  std::cout<<"KLU analysis status: "<<status<<std::endl;
  status = KLU->factorize();
  std::cout << "KLU factorization status: " << status << std::endl;
  status = KLU->solve(vec_rhs, vec_x);
  std::cout << "KLU solve status: " << status << std::endl;
  vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

  matrix_handler.setValuesChanged(true, ReSolve::memory::HOST);

  matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);

  std::cout << "\t 2-Norm of the residual: "
            << std::scientific << std::setprecision(16)
            << sqrt(vector_handler.dot(vec_r, vec_r, ReSolve::memory::HOST)) << "\n";



  //now DELETE
  delete A;
  delete KLU;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;

  return 0;
}
