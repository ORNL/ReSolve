
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#ifdef RESOLVE_USE_CUDA
  #include <resolve/LinSolverDirectCuSolverRf.hpp>
#endif
#ifdef RESOLVE_USE_HIP
  #include <resolve/LinSolverDirectRocSolverRf.hpp>
#endif
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/Profiling.hpp>
#include <resolve/utilities/params/CliOptions.hpp>

#include "ExampleHelper.hpp"

/// Prototype of the example main function 
template <class workspace_type>
static int gpuRefactor(const std::string backendName, int argc, char *argv[]);

int main(int argc, char *argv[])
{
  #ifdef RESOLVE_USE_CUDA
    return gpuRefactor<ReSolve::LinAlgWorkspaceCUDA>("CUDA", argc, argv);
  #endif
  #ifdef RESOLVE_USE_HIP
    return gpuRefactor<ReSolve::LinAlgWorkspaceHIP>("HIP", argc, argv);
  #endif
}

/*
* @brief Example usage function for the gpuRefactor
*
* @tparam workspace_type - Type of the workspace to use
* @param[in] backendName - Name of the backend to use
* @param[in] argc - Number of command line arguments
* @param[in] argv - Command line arguments
* @return 0 if the example ran successfully, -1 otherwise
*/
template <class workspace_type>
int gpuRefactor(const std::string backendName, int argc, char *argv[])
{
  std::cout << "gpuRefactor" << " with " << backendName << " backend\n";
  std::string solverName;
  if (backendName == "CUDA") {
    solverName = "CuSolver";
  } else if (backendName == "HIP")
  {
    solverName = "RocSolver";
  }
  
  using namespace ReSolve::examples;
  using namespace ReSolve;
  using index_type = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;

  const std::string example_name("gpuRefactor.exe");

  CliOptions options(argc, argv);
  CliOptions::Option* opt = nullptr;

  bool is_help = options.hasKey("-h");
  if (is_help) {
    printUsageSystemSeries(example_name);
    return 0;
  }

  index_type num_systems = 0;
  opt = options.getParamFromKey("-n");
  if (opt) {
    num_systems = atoi((opt->second).c_str());
  } else {
    std::cout << "Incorrect input!\n";
    printUsageSystemSeries(example_name);
  }

  std::string matrix_pathname("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printUsageSystemSeries(example_name);
  }

  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printUsageSystemSeries(example_name);
  }

  std::cout << "Family mtx file name: "       << matrix_pathname
            << ", total number of matrices: " << num_systems << "\n"
            << "Family rhs file name: "       << rhs_pathname
            << ", total number of RHSes: "    << num_systems << "\n";

  // Create workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Create a helper object (computing errors, printing summaries, etc.)
  ExampleHelper<workspace_type> helper(workspace);

  // Create matrix and vector handlers
  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  // Direct solvers instantiation  
  LinSolverDirectKLU KLU;
  #ifdef RESOLVE_USE_CUDA
    LinSolverDirectCuSolverRf Rf(&workspace);
  #endif
  #ifdef RESOLVE_USE_HIP
    LinSolverDirectRocSolverRf Rf(&workspace);
  #endif
  // Iterative solver instantiation
  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Pointers to matrix and vectors defining the linear system
  matrix::Csr* A = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  RESOLVE_RANGE_PUSH(__FUNCTION__);
  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "System " << i << ":\n";
    RESOLVE_RANGE_PUSH("Matrix Read");

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
      A = io::createCsrFromFile(mat_file, is_expand_symmetric);
      vec_rhs = io::createVectorFromFile(rhs_file);
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(memory::HOST);//for KLU
      vec_x->allocate(memory::DEVICE);
    } else {
      io::updateMatrixFromFile(mat_file, A);
      io::updateVectorFromFile(rhs_file, vec_rhs);
    }

    mat_file.close();
    rhs_file.close();

    // Copy data to device
    A->syncData(memory::DEVICE);
    vec_rhs->syncData(memory::DEVICE);

    printSystemInfo(matrix_pathname_full, A);
    if (!A || !vec_rhs || !vec_x) {
      std::cerr << "Null pointer encountered at iteration " << i << std::endl;
      return -1;
    }

    RESOLVE_RANGE_POP("Matrix Read");
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    int status = 0;

    if (i < 2) {
      RESOLVE_RANGE_PUSH("KLU");

      // Setup factorization solver
      KLU.setup(A);
      matrix_handler.setValuesChanged(true, memory::DEVICE);

      // Analysis (symbolic factorization)
      status = KLU.analyze();
      std::cout << "KLU analysis status: " << status << std::endl;

      // Numeric factorization
      status = KLU.factorize();
      std::cout << "KLU factorization status: " << status << std::endl;

      // Triangular solve
      status = KLU.solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;

      // Print summary of results
      helper.resetSystem(A, vec_rhs, vec_x);
      helper.printShortSummary();

      if (i == 1) {
        // Extract factors and configure refactorization solver
        matrix::Csc* L = (matrix::Csc*) KLU.getLFactor();
        matrix::Csc* U = (matrix::Csc*) KLU.getUFactor();
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        index_type* P = KLU.getPOrdering();
        index_type* Q = KLU.getQOrdering();
        if (backendName == "HIP") {
          Rf.setSolveMode(1);
        }
        Rf.setup(A, L, U, P, Q, vec_rhs);
        // Refactorization
        Rf.refactorize();

        // Setup iterative refinement solver
        FGMRES.setup(A); 
      }
      RESOLVE_RANGE_POP("KLU");
    } else {
      RESOLVE_RANGE_PUSH(solverName.c_str());
      std::cout << "Using " << solverName.c_str() << " RF\n";

      // Refactorize on the device
      status = Rf.refactorize();

      // Triangular solve on the device
      status = Rf.solve(vec_rhs, vec_x);

      // Print summary of the results
      helper.resetSystem(A, vec_rhs, vec_x);
      helper.printSummary();

      // Setup iterative refinement
      FGMRES.resetMatrix(A);
      FGMRES.setupPreconditioner("LU", &Rf);

      // If refactorization produced finite solution do iterative refinement
      if (std::isfinite(helper.getNormRelativeResidual())) {
        FGMRES.solve(vec_rhs, vec_x);

        // Print summary
        helper.printIrSummary(&FGMRES);
      }
      RESOLVE_RANGE_POP(solverName.c_str());
    }

  } // for (int i = 0; i < num_systems; ++i)
  RESOLVE_RANGE_POP(__FUNCTION__);

  delete A;
  delete vec_x;
  delete vec_rhs;

  return 0;
}
