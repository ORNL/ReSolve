/**
 * @file sysGmres.cpp
 * @author Kakeru Ueda (ueda.k.2290@m.isct.ac.jp)
 * @brief Example of solving a linear system with GMRES using SystemSolver.
 */

#include <iomanip>
#include <iostream>
#include <string>

#include "ExampleHelper.hpp"
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

// Uses ReSolve data types
using namespace ReSolve;
using namespace ReSolve::examples;
using namespace ReSolve::memory;
using vector_type = ReSolve::vector::Vector;

/// Prints help message describing system usage
void printHelpInfo()
{
  std::cout << "\nsysGmres.exe loads a linear system from files and solves it using GMRES.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "sysGmres.exe -m <matrix file> -r <rhs file>\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-b <cpu|cuda|hip> \tSelects hardware backend.\n";
  std::cout << "\t-h \tPrints this message.\n";
  std::cout << "\t-i <iter method> \tIterative method: randgmres or fgmres (default 'randgmres').\n";
  std::cout << "\t-g <gs method> \tGram-Schmidt method: cgs1, cgs2, or mgs (default 'cgs2').\n";
  std::cout << "\t-s <sketching method> \tSketching method: count or fwht (default 'count')\n";
  std::cout << "\t-x <flexible> \tEnable flexible: yes or no (default 'yes')\n\n";
}

//
// Forward declarations of functions
//

/**
 * @brief Example of solving a linear system with GMRES using SystemSolver.
 *
 * @tparam workspace_type - Type of the workspace to use
 * @param[in] argc - Number of command line arguments
 * @param[in] argv - Command line arguments
 * @return 0 if the example ran successfully, 1 otherwise
 */
template <class workspace_type>
static int sysGmres(int argc, char* argv[]);

/// Checks if inputs for GMRES are valid, otherwise sets defaults
static void processInputs(std::string& method,
                          std::string& gs,
                          std::string& sketch,
                          std::string& flexible);

/// Main function selects example to be run
int main(int argc, char* argv[])
{
  CliOptions options(argc, argv);

  bool is_help = options.hasKey("-h");
  if (is_help)
  {
    printHelpInfo();
    return 0;
  }

  auto opt = options.getParamFromKey("-b");
  if (!opt)
  {
    std::cout << "No backend option provided. Defaulting to CPU.\n";
    return sysGmres<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  }
#ifdef RESOLVE_USE_CUDA
  else if (opt->second == "cuda")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceCUDA>(argc, argv);
  }
#endif
#ifdef RESOLVE_USE_HIP
  else if (opt->second == "hip")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
  }
#endif
  else if (opt->second == "cpu")
  {
    return sysGmres<ReSolve::LinAlgWorkspaceCpu>(argc, argv);
  }
  else
  {
    std::cout << "Re::Solve is not build with support for " << opt->second;
    std::cout << " backend.\n";
    return 1;
  }

  return 0;
}

//
// Definitions of functions
//

template <class workspace_type>
int sysGmres(int argc, char* argv[])
{
  // return_code is used as a failure flag.
  int return_code = 0;
  int status      = 0;

  // Collect all CLI
  CliOptions options(argc, argv);

  bool is_help = options.hasKey("-h");
  if (is_help)
  {
    printHelpInfo();
    return 0;
  }

  // Read matrix file
  auto        opt = options.getParamFromKey("-m");
  std::string matrix_pathname("");
  if (opt)
  {
    matrix_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  }

  // Read right-hand-side vector file
  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt)
  {
    rhs_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  }

  // Read GMRES-related options
  opt                = options.getParamFromKey("-i");
  std::string method = opt ? (*opt).second : "randgmres";

  opt            = options.getParamFromKey("-g");
  std::string gs = opt ? (*opt).second : "cgs2";

  opt                = options.getParamFromKey("-s");
  std::string sketch = opt ? (*opt).second : "count";

  opt                  = options.getParamFromKey("-x");
  std::string flexible = opt ? (*opt).second : "yes";

  processInputs(method, gs, sketch, flexible);

  std::cout << "Matrix file: " << matrix_pathname << "\n"
            << "RHS file: " << rhs_pathname << "\n";

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);
  std::string                   hw_backend = helper.getHardwareBackend();
  std::cout << "sysGmres with " << hw_backend << " backend\n";

  // Set memory space
  MemorySpace memspace = helper.getMemspace();

  // Set solver
  SystemSolver solver(&workspace,
                      "none",
                      "none",
                      method,
                      "ilu0",
                      "none");

  solver.setGramSchmidtMethod(gs);

  // Read and open matrix and right-hand-side vector
  std::ifstream mat_file(matrix_pathname);
  if (!mat_file.is_open())
  {
    std::cout << "Failed to open matrix file: " << matrix_pathname << "\n";
    return 1;
  }
  std::ifstream rhs_file(rhs_pathname);
  if (!rhs_file.is_open())
  {
    std::cout << "Failed to open RHS file: " << rhs_pathname << "\n";
    return 1;
  }

  bool is_expand_symmetric = true;

  // Load system matrix and RHS vector from input files
  matrix::Csr* A       = io::createCsrFromFile(mat_file, is_expand_symmetric);
  vector_type* vec_rhs = io::createVectorFromFile(rhs_file);

  // Create solution vector
  vector_type* vec_x = new vector_type(A->getNumRows());
  vec_x->allocate(memspace);

  if (memspace == memory::DEVICE)
  {
    // Copy data to the device
    A->syncData(memspace);
    vec_rhs->syncData(memspace);
  }

  mat_file.close();
  rhs_file.close();

  // Set iterative solver options
  solver.getIterativeSolver().setCliParam("maxit", "2500");
  solver.getIterativeSolver().setCliParam("tol", "1e-12");

  status = solver.setMatrix(A);
  std::cout << "solver.setMatrix returned status: " << status << "\n";
  if (status != 0)
  {
    return_code = 1;
  }

  // Set GMRES solver options
  if (method == "randgmres")
  {
    solver.setSketchingMethod(sketch);
  }
  solver.getIterativeSolver().setCliParam("flexible", flexible);
  solver.getIterativeSolver().setCliParam("restart", "200");

  // Set up the preconditioner
  if (return_code == 0)
  {
    status = solver.preconditionerSetup();
    std::cout << "solver.preconditionerSetup returned status: " << status << "\n";
    if (status != 0)
    {
      return_code = 1;
    }
  }

  // Solve the system
  if (return_code == 0)
  {
    status = solver.solve(vec_rhs, vec_x);
    std::cout << "solver.solve returned status: " << status << "\n";
    if (status != 0)
    {
      return_code = 1;
    }
  }

  if (return_code == 0)
  {
    // Get reference to iterative solver and print results
    LinSolverIterative& iter_solver = solver.getIterativeSolver();
    helper.printIterativeSolverSummary(&iter_solver);
  }

  // Free matrix/vector objects.
  delete A;
  delete vec_rhs;
  delete vec_x;

  return return_code;
}

void processInputs(std::string& method, std::string& gs, std::string& sketch, std::string& flexible)
{
  if (method == "randgmres")
  {
    if ((sketch != "count") && (sketch != "fwht"))
    {
      std::cout << "Sketching method " << sketch << " not recognized.\n";
      std::cout << "Setting sketch to the default (count).\n\n";
      sketch = "count";
    }
  }

  if ((method != "randgmres") && (method != "fgmres"))
  {
    std::cout << "Iterative method " << method << " not recognized.\n";
    std::cout << "Setting iterative solver method to the default (RANDGMRES).\n\n";
    method = "randgmres";
  }

  if (gs != "cgs1" && gs != "cgs2" && gs != "mgs" && gs != "mgs_two_sync"
      && gs != "mgs_pm")
  {
    std::cout << "Orthogonalization method " << gs << " not recognized.\n";
    std::cout << "Setting orthogonalization to the default (CGS2).\n\n";
    gs = "cgs2";
  }

  if ((flexible != "yes") && (flexible != "no"))
  {
    std::cout << "Flexible option " << flexible << " not recognized.\n";
    std::cout << "Setting flexible to the default (yes).\n\n";
    flexible = "yes";
  }
}
