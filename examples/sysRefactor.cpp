#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string>
#include <resolve/Profiling.hpp>
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
#include <resolve/utilities/params/CliOptions.hpp>

#include "ExampleHelper.hpp"

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\nLoads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering.\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "sysRefactor.exe -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-b <cpu|cuda|hip> \tSelects hardware backend.\n";
  std::cout << "\t-e <ext> \tSelects custom extension for input files (default 'mtx').\n";
  std::cout << "\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

using namespace ReSolve::constants;

/// Prototype of the example function
template <class workspace_type>
static int sysRefactor(int argc, char *argv[]);

/// Main function selects example to be run.
int main(int argc, char *argv[])
{
  ReSolve::CliOptions options(argc, argv);

  // If help flag is passed, print help message and return
  bool is_help = options.hasKey("-h");
  if (is_help) {
    printHelpInfo();
    return 0;
  }

  // Select hardware backend, default to CPU if no -b option is passed
  ReSolve::CliOptions::Option* opt = options.getParamFromKey("-b");
  if (!opt)
  {
    std::cout << "No backend option provided. Defaulting to CPU.\n";
    sysRefactor<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
    return 0;
  }
#ifdef RESOLVE_USE_CUDA
  else if (opt->second == "cuda")
  {
    sysRefactor<ReSolve::LinAlgWorkspaceCUDA>(argc, argv);
  }
#endif
#ifdef RESOLVE_USE_HIP
  else if (opt->second == "hip")
  {
    sysRefactor<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
  }
#endif
  else if (opt->second == "cpu")
  {
    sysRefactor<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  }
  else
  {
    std::cout << "Re::Solve is not built with support for " << opt->second;
    std::cout << "backend.\n";
    return 1;
  }

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

  std::string file_extension("");
  opt = options.getParamFromKey("-e");
  if (opt) {
    file_extension = opt->second;
  } else {
    file_extension = "mtx";
  }

  std::cout << "Family matrix file name: "    << matrix_pathname
            << ", total number of matrices: " << num_systems << "\n"
            << "Family rhs file name: "       << rhs_pathname
            << ", total number of RHSes: "    << num_systems << "\n";

  int status = 0;

  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);
  std::string hw_backend = helper.getHardwareBackend();
  std::cout << "sysRefactor with " << hw_backend << " backend\n";

  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  // Pointers to the linear system
  matrix::Csr* A       = nullptr; // Pointer to the system matrix
  vector_type* vec_rhs = nullptr; // Pointer for the right hand side vector
  vector_type* vec_x   = nullptr; // Pointer for the solution vector

  // Create system solver
  std::string refactor("none");
  if (hw_backend == "CUDA") {
    refactor = "glu";
  } else if (hw_backend == "HIP") {
    refactor = "rocsolverrf";
  } else {
    refactor = "klu";
  }

  ReSolve::SystemSolver solver(&workspace,
                               "klu",    // factorization
                               refactor, // refactorization
                               refactor, // triangular solve
                               "none",   // preconditioner (always 'none' here)
                               "none");  // iterative refinement

  // Disable iterative refinement temporarily for CPU backend
  if (hw_backend == "CPU") {
    is_iterative_refinement = false;
  }

  if (is_iterative_refinement) {
    solver.setRefinementMethod("fgmres", "cgs2");
    solver.getIterativeSolver().setCliParam("restart", "100");
    if (hw_backend == "CUDA") {
      solver.getIterativeSolver().setTol(1e-17);
    }
  }

  RESOLVE_RANGE_PUSH(__FUNCTION__);
  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "System " << i << ":\n";
    RESOLVE_RANGE_PUSH("File input");
    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_pathname << std::setfill('0') << std::setw(2) << i << "." << file_extension;
    rhsname << rhs_pathname    << std::setfill('0') << std::setw(2) << i << "." << file_extension;
    std::string matrix_pathname_full = matname.str();
    std::string rhs_pathname_full    = rhsname.str();

    // Read matrix and right-hand-side vector
    std::ifstream mat_file(matrix_pathname_full);
    if(!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrix_pathname_full << "\n";
      return 1;
    }
    std::ifstream rhs_file(rhs_pathname_full);
    if(!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhs_pathname_full << "\n";
      return 1;
    }

    // Refactorization is LU-based, so need to expand symmetric matrices
    bool is_expand_symmetric = true;
    if (i == 0) {
      A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);
      vec_rhs = ReSolve::io::createVectorFromFile(rhs_file);
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(memory::HOST);
      if (hw_backend == "CUDA" || hw_backend == "HIP") {
        vec_x->allocate(memory::DEVICE);
      }
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateVectorFromFile(rhs_file, vec_rhs);
    }

    mat_file.close();
    rhs_file.close();

    // Ensure matrix data is synced to the device before any GPU operations
    if (hw_backend == "CUDA" || hw_backend == "HIP") {
      A->syncData(memory::DEVICE);
      vec_rhs->syncData(memory::DEVICE);
    }

    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;
    printSystemInfo(matrix_pathname_full, A);

    // Now call direct solver
    if (i == 0) {
      // Set matrix in solver after the initial matrix is loaded
      status = solver.setMatrix(A);
      if (status != 0) {
        std::cout << "Failed to set matrix in solver. Status: " << status << std::endl;
        return 1;
      }

      // Analysis (symbolic factorization)
      status = solver.analyze();
      std::cout << "Analysis on the host status: " << status << std::endl;

      // Numeric factorization on the host
      status = solver.factorize();
      std::cout << "Numeric factorization on the host status: " << status << std::endl;
    } else if (i == 1) {
      // Numeric factorization on the host
      status = solver.factorize();
      std::cout << "Numeric factorization on the host status: " << status << std::endl;

      // Set up refactorization solver
      status = solver.refactorizationSetup();
      std::cout << "Refactorization setup status: " << status << std::endl;

    } else {
      // Refactorize on the device
      status = solver.refactorize();
      std::cout << "Refactorization on the device status: " << status << std::endl;
    }

    status = solver.solve(vec_rhs, vec_x);
    std::cout << "Triangular solve status: " << status << std::endl;

    // Print summary of results
    helper.resetSystem(A, vec_rhs, vec_x);
    helper.printShortSummary();
    if ((i > 1) && is_iterative_refinement) {
      helper.printIrSummary(&(solver.getIterativeSolver()));
    }
  }

  // Delete objects created on heap
  delete A;
  delete vec_x;   // Delete the solution vector
  delete vec_rhs; // Delete the RHS vector

  return 0;
}