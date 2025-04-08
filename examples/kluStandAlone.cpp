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
#include <resolve/utilities/params/CliOptions.hpp>

using namespace ReSolve::constants;

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\nLoads from files and solves a linear system.\n\n";
  std::cout << "System matrix is in a file with name <matrix file name>\n";
  std::cout << "System right hand side vector is stored in a file with name <rhs file name>\n\n";
  std::cout << "kluStandAlone.exe -m <matrix file name> -r <rhs file name> \n\n";
  std::cout << "Optional features:\n\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  ReSolve::CliOptions options(argc, argv);
  ReSolve::CliOptions::Option* opt = nullptr;

  bool is_help = options.hasKey("-h");
  if (is_help) {
    printHelpInfo();
    return 0;
  }

  bool is_iterative_refinement = options.hasKey("-i");

  std::string matrix_file_name("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_file_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string rhs_file_name("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_file_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  ReSolve::matrix::Csr* A = nullptr;
  ReSolve::LinAlgWorkspaceCpu* workspace = new ReSolve::LinAlgWorkspaceCpu();
  ReSolve::MatrixHandler* matrix_handler = new ReSolve::MatrixHandler(workspace);
  ReSolve::VectorHandler* vector_handler = new ReSolve::VectorHandler(workspace);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  vector_type* vec_r   = nullptr;

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  // Create a helper object (computing errors, printing summaries, etc.)
  ReSolve::examples::ExampleHelper<ReSolve::LinAlgWorkspaceCpu> helper(*workspace);

  std::ifstream mat_file(matrix_file_name);
  if(!mat_file.is_open())
  {
    std::cout << "Failed to open file " << matrix_file_name<< "\n";
    return -1;
  }
  std::ifstream rhs_file(rhs_file_name);
  if(!rhs_file.is_open())
  {
    std::cout << "Failed to open file " << rhs_file_name << "\n";
    return -1;
  }
  bool is_expand_symmetric = true;
  A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

  rhs = ReSolve::io::createArrayFromFile(rhs_file);
  x = new real_type[A->getNumRows()];
  vec_rhs = new vector_type(A->getNumRows());
  vec_x = new vector_type(A->getNumRows());
  vec_r = new vector_type(A->getNumRows());
  helper.resetSystem(A, vec_rhs, vec_x);
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

  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

  matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);

  helper.printShortSummary();

  //now DELETE
  delete A;
  delete KLU;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
