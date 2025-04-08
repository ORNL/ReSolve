#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>
#include "ExampleHelper.hpp"
#include <resolve/utilities/params/CliOptions.hpp>

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\nLoads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering.\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "gpuRefactor.exe -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}


using namespace ReSolve::constants;

/// Prototype of the example function
template <class workspace_type>
static int sysRefactor(int argc, char *argv[]);

/// Main function selects example to be run.
int main(int argc, char *argv[])
{
  sysRefactor<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  #ifdef RESOLVE_USE_CUDA
    sysRefactor<ReSolve::LinAlgWorkspaceCUDA>(argc, argv);
  #endif

  #ifdef RESOLVE_USE_HIP
    sysRefactor<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
  #endif

  return 0;
}

/**
 * @brief Example of using system solvers on GPU
 *
 * @tparam workspace_type - Type of the workspace to use
 * @param[in] argc - Number of command line arguments
 * @param[in] argv - Command line arguments
 * @return 0 if the example ran successfully, -1 otherwise
 */
template <class workspace_type>
int sysRefactor(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using namespace ReSolve::examples;
  using namespace ReSolve;
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  CliOptions options(argc, argv);
  CliOptions::Option* opt = nullptr;

  bool is_help = options.hasKey("-h");
  if (is_help) {
    printHelpInfo();
    return 0;
  }

  bool is_iterative_refinement = options.hasKey("-i");

  index_type num_systems = 0;
  opt = options.getParamFromKey("-n");
  if (opt) {
    num_systems = atoi((opt->second).c_str());
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string matrix_pathname("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::cout << "Family mtx file name: "       << matrix_pathname
            << ", total number of matrices: " << num_systems << "\n"
            << "Family rhs file name: "       << rhs_pathname
            << ", total number of RHSes: "    << num_systems << "\n";

  std::string filed_id;
  std::string rhs_id;
  std::string matrix_file_name_full;
  std::string rhs_file_name_full;

  matrix::Csr* A = nullptr;
  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);
  std::cout << "sysRefactor with " << helper.getHardwareBackend() << " backend\n";

  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  vector_type* vec_r   = nullptr;

  SystemSolver* solver = new SystemSolver(&workspace);
  solver->setRefinementMethod("fgmres", "cgs2");

  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "System " << i << ":\n";
    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_pathname << std::setfill('0') << std::setw(2) << i << ".mtx";
    rhsname << rhs_pathname    << std::setfill('0') << std::setw(2) << i << ".mtx";
    std::string matrix_pathname_full = matname.str();
    std::string rhs_pathname_full    = rhsname.str();

    // Read matrix and right-hand-side vector
    std::ifstream mat_file(matrix_pathname_full);
    if(!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrix_pathname_full << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhs_pathname_full);
    if(!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhs_pathname_full << "\n";
      return -1;
    }
    bool is_expand_symmetric = true;
    if (i == 0) {
      A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

      rhs = ReSolve::io::createArrayFromFile(rhs_file);
      x = new real_type[A->getNumRows()];
      vec_rhs = new vector_type(A->getNumRows());
      vec_x = new vector_type(A->getNumRows());
      vec_r = new vector_type(A->getNumRows());
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
    }
    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows()
              << " x "           << A->getNumColumns()
              << ", nnz: "       << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? "  << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    // Update data.
    if (i < 2) {
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else {
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    }
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnz()<<std::endl;
    //Now call direct solver
    solver->setMatrix(A);
    int status;
    if (i < 2){
      // solver->setup(A);
      status = solver->analyze();
      std::cout<<"solver analysis status: "<<status<<std::endl;
      status = solver->factorize();
      std::cout<<"solver factorization status: "<<status<<std::endl;
      status = solver->solve(vec_rhs, vec_x);
      std::cout<<"solver solve status: "<<status<<std::endl;
    } else {
      status =  solver->refactorize();
      std::cout<<"solver re-factorization status: "<<status<<std::endl;
      status = solver->solve(vec_rhs, vec_x);
      std::cout<<"solver solve status: "<<status<<std::endl;
    }
    vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler.setValuesChanged(true, ReSolve::memory::HOST);

    matrix_handler.matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);

    // Print summary of results
    helper.resetSystem(A, vec_rhs, vec_x);
    helper.printShortSummary();

  }

  //now DELETE
  delete A;
  delete solver;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete vec_rhs;

  return 0;
}
