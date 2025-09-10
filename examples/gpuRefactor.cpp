/**
 * @file gpuRefactor.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @author Kasia Swirydowicz (kasia.swirydowicz@amd.com)
 *
 * @brief Example of solving linear systems using refactorization on a GPU.
 *
 * A series of linear systems is read from files specified at command line
 * input and solved with refactorization approach on GPU. Initially, system(s)
 * are solved with KLU solver on CPU, using full factorization, and the
 * subsequent systems are solved with refactorization solver on GPU. If the
 * example is built with CUDA, cusolverRf is used for refactorization. For HIP
 * builds, rocsolverRf is used. It is assumed that all systems in the series
 * have the same sparsity pattern, so the analysis is done only once for the
 * entire series.
 *
 */
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/Profiling.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#endif
#ifdef RESOLVE_USE_HIP
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif

#include "ExampleHelper.hpp"

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\ngpuRefactor.exe loads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "gpuRefactor.exe -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

/// Prototype of the example function
template <class workspace_type, class refactor_type>
static int gpuRefactor(int argc, char* argv[]);

/// Main function selects example to be run.
int main(int argc, char* argv[])
{
#ifdef RESOLVE_USE_CUDA
  gpuRefactor<ReSolve::LinAlgWorkspaceCUDA,
              ReSolve::LinSolverDirectCuSolverRf>(argc, argv);
#endif

#ifdef RESOLVE_USE_HIP
  gpuRefactor<ReSolve::LinAlgWorkspaceHIP,
              ReSolve::LinSolverDirectRocSolverRf>(argc, argv);
#endif

  return 0;
}

/**
 * @brief Example of using refactorization solvers on GPU
 *
 * @tparam workspace_type - Type of the workspace to use
 * @param[in] argc - Number of command line arguments
 * @param[in] argv - Command line arguments
 * @return 0 if the example ran successfully, -1 otherwise
 */
template <class workspace_type, class refactor_type>
int gpuRefactor(int argc, char* argv[])
{
  using namespace ReSolve::examples;
  using namespace ReSolve;
  using index_type  = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;

  CliOptions options(argc, argv);

  bool is_help = options.hasKey("-h");
  if (is_help)
  {
    printHelpInfo();
    return 0;
  }

  bool is_iterative_refinement = options.hasKey("-i");

  index_type num_systems = 0;
  auto       opt         = options.getParamFromKey("-n");
  if (opt)
  {
    num_systems = atoi((opt->second).c_str());
  }
  else
  {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string matrix_pathname("");
  opt = options.getParamFromKey("-m");
  if (opt)
  {
    matrix_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt)
  {
    rhs_pathname = opt->second;
  }
  else
  {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string file_extension("");
  opt = options.getParamFromKey("-e");
  if (opt)
  {
    file_extension = opt->second;
  }
  else
  {
    file_extension = "mtx";
  }

  std::cout << "Family mtx file name: " << matrix_pathname
            << ", total number of matrices: " << num_systems << "\n"
            << "Family rhs file name: " << rhs_pathname
            << ", total number of RHSes: " << num_systems << "\n";

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);
  std::cout << "gpuRefactor with " << helper.getHardwareBackend() << " backend\n";

  // Create matrix and vector handlers
  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  // Direct solvers instantiation
  LinSolverDirectKLU KLU;
  refactor_type      Rf(&workspace);

  // Iterative solver instantiation
  GramSchmidt              GS(&vector_handler, GramSchmidt::CGS2);
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Pointers to matrix and vectors defining the linear system
  matrix::Csr* A       = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  RESOLVE_RANGE_PUSH(__FUNCTION__);
  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "System " << i << ":\n";

    RESOLVE_RANGE_PUSH("File input");
    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_pathname << std::setfill('0') << std::setw(2) << i << "." << file_extension;
    rhsname << rhs_pathname << std::setfill('0') << std::setw(2) << i << "." << file_extension;
    std::string matrix_pathname_full = matname.str();
    std::string rhs_pathname_full    = rhsname.str();

    // Read matrix and right-hand-side vector
    std::ifstream mat_file(matrix_pathname_full);
    if (!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrix_pathname_full << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhs_pathname_full);
    if (!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhs_pathname_full << "\n";
      return -1;
    }
    bool is_expand_symmetric = true;
    if (i == 0)
    {
      A       = io::createCsrFromFile(mat_file, is_expand_symmetric);
      vec_rhs = io::createVectorFromFile(rhs_file);
      vec_x   = new vector_type(A->getNumRows());
      vec_x->allocate(memory::HOST);
      vec_x->allocate(memory::DEVICE);
    }
    else
    {
      io::updateMatrixFromFile(mat_file, A);
      io::updateVectorFromFile(rhs_file, vec_rhs);
    }

    mat_file.close();
    rhs_file.close();

    // Copy data to device
    A->syncData(memory::DEVICE);
    vec_rhs->syncData(memory::DEVICE);
    RESOLVE_RANGE_POP("File input");

    printSystemInfo(matrix_pathname_full, A);
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    int status = 0;

    if (i == 0)
    {
      RESOLVE_RANGE_PUSH("KLU");
      // Setup factorization solver
      KLU.setup(A);
      matrix_handler.setValuesChanged(true, memory::DEVICE);

      // Analysis (symbolic factorization)
      status = KLU.analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
    }

    if (i < 2)
    {
      // Numeric factorization
      status = KLU.factorize();
      std::cout << "KLU factorization status: " << status << std::endl;

      // Triangular solve
      status = KLU.solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;

      // Print summary of results
      helper.resetSystem(A, vec_rhs, vec_x);
      helper.printShortSummary();

      if (i == 1)
      {
        // Extract factors and configure refactorization solver
        matrix::Csr* L = (matrix::Csr*) KLU.getLFactorCsr();
        matrix::Csr* U = (matrix::Csr*) KLU.getUFactorCsr();
        if (L == nullptr || U == nullptr)
        {
          std::cout << "Factor extraction from KLU failed!\n";
        }
        index_type* P = KLU.getPOrdering();
        index_type* Q = KLU.getQOrdering();

        Rf.setupCsr(A, L, U, P, Q, vec_rhs);

        // Setup iterative refinement solver
        if (is_iterative_refinement)
        {
          FGMRES.setup(A);
        }
      }
      RESOLVE_RANGE_POP("KLU");
    }
    else
    {
      std::cout << "Using refactorization\n";

      RESOLVE_RANGE_PUSH("Refactorization");
      // Refactorize on the device
      status = Rf.refactorize();

      // Triangular solve on the device
      status = Rf.solve(vec_rhs, vec_x);
      RESOLVE_RANGE_POP("Refactorization");

      // Print summary of the results
      helper.resetSystem(A, vec_rhs, vec_x);
      helper.printSummary();

      RESOLVE_RANGE_PUSH("Iterative refinement");
      if (is_iterative_refinement)
      {
        // Setup iterative refinement
        FGMRES.resetMatrix(A);
        FGMRES.setupPreconditioner("LU", &Rf);

        // If refactorization produced finite solution do iterative refinement
        if (std::isfinite(helper.getNormRelativeResidual()))
        {
          FGMRES.solve(vec_rhs, vec_x);

          // Print summary
          helper.printIrSummary(&FGMRES);
        }
      }
      RESOLVE_RANGE_POP("Iterative refinement");
    }

  } // for (int i = 0; i < num_systems; ++i)
  RESOLVE_RANGE_POP(__FUNCTION__);

  delete A;
  delete vec_x;
  delete vec_rhs;

  return 0;
}
