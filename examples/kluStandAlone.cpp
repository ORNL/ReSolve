#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include "ExampleHelper.hpp"
#include <resolve/utilities/params/CliOptions.hpp>

using namespace ReSolve::constants;

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\nLoads from files and solves a linear system.\n\n";
  std::cout << "Usage:\n\t./kluStandAlone.exe -m <matrix pathname> -r <rhs pathname>\n\n";
  std::cout << "Optional features:\n\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

int main(int argc, char *argv[])
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using namespace ReSolve;
  using namespace ReSolve::examples;

  CliOptions options(argc, argv);
  CliOptions::Option* opt = nullptr;

  bool is_help = options.hasKey("-h");
  if (is_help) {
    printHelpInfo();
    return 0;
  }

  bool is_iterative_refinement = options.hasKey("-i");

  std::string matrix_path_name("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  }

  std::string rhs_path_name("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
    return 1;
  }

  matrix::Csr* A = nullptr;
  LinAlgWorkspaceCpu workspace;
  ExampleHelper<LinAlgWorkspaceCpu> helper(workspace);
  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x = nullptr;

  LinSolverDirectKLU* KLU = new LinSolverDirectKLU;

  // Load the system
  std::cout << "Solving the system:\n";

  std::ifstream mat_file(matrix_path_name);
  if(!mat_file.is_open())
  {
    std::cout << "Failed to open file " << matrix_path_name << "\n";
    return 1;
  }

  std::ifstream rhs_file(rhs_path_name);
  if(!rhs_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_path_name << "\n";
    return 1;
  }

  bool is_expand_symmetric = true;
  A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);
  vec_rhs = ReSolve::io::createVectorFromFile(rhs_file);
  vec_x = new vector_type(A->getNumRows());

  printSystemInfo(matrix_path_name, A);
  mat_file.close();
  rhs_file.close();

  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  std::cout << "COO to CSR completed. Expanded NNZ: " << A->getNnz() << std::endl;

  // Direct solver
  int status;
  KLU->setup(A);
  status = KLU->analyze();
  std::cout << "KLU analysis status: " << status << std::endl;
  status = KLU->factorize();
  std::cout << "KLU factorization status: " << status << std::endl;

  // Solve the system
  status = KLU->solve(vec_rhs, vec_x);
  std::cout << "KLU solve status: " << status << std::endl;

  helper.resetSystem(A, vec_rhs, vec_x);
  helper.printShortSummary();
  if (is_iterative_refinement) {
    GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);
    LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);
    // Setup iterative refinement
    FGMRES.setup(A);
    FGMRES.setupPreconditioner("LU", KLU);
    // If refactorization produced finite solution do iterative refinement
    if (std::isfinite(helper.getNormRelativeResidual())) {
      FGMRES.solve(vec_rhs, vec_x);
      helper.printIrSummary(&FGMRES);
    }
  }

  // Cleanup
  delete A;
  delete KLU;
  delete vec_rhs;
  delete vec_x;
  return 0;
}