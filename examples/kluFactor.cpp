/**
 * @file kluFactor.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @author Kasia Swirydowicz (kasia.swirydowicz@amd.com)

 * @brief Example solving linear systems using KLU factorization
 * 
 * A series of linear systems is read from files specified at command line
 * input and solved with KLU solver, using full factorization for each
 * system. It is assumed that all systems in the series have the same sparsity
 * pattern, so the analysis is done only once for the entire series.
 * 
 */
#include <iostream>
#include <iomanip>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include "ExampleHelper.hpp"
#include <resolve/utilities/params/CliOptions.hpp>

using namespace ReSolve::constants;

/// Prints help message describing system usage.
void printHelpInfo()
{
  std::cout << "\nkluFactor.exe loads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./";
  std::cout << "kluFactor.exe -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n";
  std::cout << "\t-h\tPrints this message.\n";
  std::cout << "\t-i\tEnables iterative refinement.\n\n";
}

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;
  using namespace ReSolve::examples;
  using namespace ReSolve;
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
    num_systems = std::stoi((opt->second).c_str());
  } else {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string matrix_path_name("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string rhs_path_name("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_path_name = opt->second;
  } else {
    std::cout << "Incorrect input!\n\n";
    printHelpInfo();
    return 1;
  }

  std::string fileId;
  std::string rhsId;
  std::string matrix_file_name_full;
  std::string rhs_file_name_full;

  matrix::Csr* A = nullptr;
  LinAlgWorkspaceCpu workspace;
  ExampleHelper<LinAlgWorkspaceCpu> helper(workspace);
  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  LinSolverDirectKLU KLU;
  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);
  for (int i = 0; i < num_systems; ++i)
  {
    std::cout << "System " << i << ":\n";

    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_path_name << std::setfill('0') << std::setw(2) << i << ".mtx";
    rhsname << rhs_path_name    << std::setfill('0') << std::setw(2) << i << ".mtx";
    matrix_file_name_full = matname.str();
    rhs_file_name_full = rhsname.str();
    std::ifstream mat_file(matrix_file_name_full);
    if(!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrix_file_name_full << "\n";
      return 1;
    }
    std::ifstream rhs_file(rhs_file_name_full);
    if(!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhs_file_name_full << "\n";
      return 1;
    }
    bool is_expand_symmetric = true;
    if (i == 0) {
      A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

      vec_rhs = ReSolve::io::createVectorFromFile(rhs_file);
      vec_x = new vector_type(A->getNumRows());
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateVectorFromFile(rhs_file, vec_rhs);
    }
    printSystemInfo(matrix_file_name_full, A);
    mat_file.close();
    rhs_file.close();

    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnz()<<std::endl;
    //Now call direct solver
    int status;
    if (i==0) {
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
      KLU.setup(A);
      status = KLU.analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
    }

    status = KLU.factorize();
    std::cout<<"KLU factorization status: "<<status<<std::endl;

    status = KLU.solve(vec_rhs, vec_x);
    std::cout<<"KLU solve status: "<<status<<std::endl;

    helper.resetSystem(A, vec_rhs, vec_x);
    helper.printShortSummary();
    if (is_iterative_refinement) {
      // Setup iterative refinement
      FGMRES.setup(A);
      FGMRES.setupPreconditioner("LU", &KLU);

      // If refactorization produced finite solution do iterative refinement
      if (std::isfinite(helper.getNormRelativeResidual())) {
        FGMRES.solve(vec_rhs, vec_x);

        // Print summary
        helper.printIrSummary(&FGMRES);
      }
    }
  }

  //now DELETE
  delete A;
  delete vec_rhs;
  delete vec_x;
  return 0;
}
