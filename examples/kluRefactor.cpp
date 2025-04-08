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
  std::cout << "\nLoads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering.\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "kluRefactor.exe -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
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

  index_type num_systems = 0;
  opt = options.getParamFromKey("-n");
  if (opt) {
    num_systems = std::stoi((opt->second).c_str());
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string matrix_path_name("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string rhs_path_name("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printHelpInfo();
  }

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::matrix::Csr* A = nullptr;
  ReSolve::LinAlgWorkspaceCpu workspace;
  ReSolve::examples::ExampleHelper<ReSolve::LinAlgWorkspaceCpu> helper(workspace);
  ReSolve::MatrixHandler* matrix_handler = new ReSolve::MatrixHandler(&workspace);
  ReSolve::VectorHandler* vector_handler = new ReSolve::VectorHandler(&workspace);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  vector_type* vec_r   = nullptr;
  real_type norm_A, norm_x, norm_r;//used for INF norm

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

  for (int i = 0; i < num_systems; ++i)
  {
    index_type j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrix_path_name + fileId + ".mtx";
    rhsFileNameFull = rhs_path_name  + rhsId + ".mtx";
    helper.resetSystem(A, vec_rhs, vec_x);
    helper.printShortSummary();
    std::ifstream mat_file(matrixFileNameFull);
    if(!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrixFileNameFull << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhsFileNameFull);
    if(!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhsFileNameFull << "\n";
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
    int status;

    if (i < 2){
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
    } else {
      status =  KLU->refactorize();
      std::cout<<"KLU re-factorization status: "<<status<<std::endl;
    }
    status = KLU->solve(vec_rhs, vec_x);
    std::cout<<"KLU solve status: "<<status<<std::endl;
    vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::HOST);
    norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::HOST);

    std::cout << "\t2-Norm of the residual: "
              << std::scientific << std::setprecision(16)
              << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST)) << "\n";
    matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::HOST);
    norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::HOST);
    std::cout << "\tMatrix inf  norm: " << std::scientific << std::setprecision(16) << norm_A<<"\n"
              << "\tResidual inf norm: " << norm_r <<"\n"
              << "\tSolution inf norm: " << norm_x <<"\n"
              << "\tNorm of scaled residuals: "<< norm_r / (norm_A * norm_x) << "\n";
  }

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
