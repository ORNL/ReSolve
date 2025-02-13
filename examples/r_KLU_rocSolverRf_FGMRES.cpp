#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectRocSolverRf.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/Profiling.hpp>
#include <resolve/utilities/params/CliOptions.hpp>

#include "ExampleHelper.hpp"

void printUsage(const std::string& name)
{
  std::cout << "\nLoads from files and solves a series of linear systems.\n\n";
  std::cout << "System matrices are in files with names <pathname>XX.mtx, where XX are\n";
  std::cout << "consecutive integer numbers 00, 01, 02, ...\n\n";
  std::cout << "System right hand side vectors are stored in files with matching numbering.\n";
  std::cout << "and file extension.\n\n";
  std::cout << "Usage:\n\t./" << name;
  std::cout << " -m <matrix pathname> -r <rhs pathname> -n <number of systems>\n\n";
  std::cout << "Optional features:\n\t-h\tPrints this message.\n\n";
}

template <class workspace_type>
static int example(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  return example<ReSolve::LinAlgWorkspaceHIP>(argc, argv);
}

template <class workspace_type>
int example(int argc, char *argv[])
{
  using namespace ReSolve::examples;
  using namespace ReSolve;
  using index_type = ReSolve::index_type;
  using vector_type = ReSolve::vector::Vector;

  const std::string example_name("klu_rocsolverrf_fgmres.exe");

  CliOptions options(argc, argv);
  CliOptions::Option* opt = nullptr;

  bool is_help = options.hasKey("-h");
  if (is_help) {
    printUsage(example_name);
    return 0;
  }

  index_type num_systems = 0;
  opt = options.getParamFromKey("-n");
  if (opt) {
    num_systems = atoi((opt->second).c_str());
  } else {
    std::cout << "Incorrect input!\n";
    printUsage(example_name);
  }

  std::string matrix_pathname("");
  opt = options.getParamFromKey("-m");
  if (opt) {
    matrix_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printUsage(example_name);
  }

  std::string rhs_pathname("");
  opt = options.getParamFromKey("-r");
  if (opt) {
    rhs_pathname = opt->second;
  } else {
    std::cout << "Incorrect input!\n";
    printUsage(example_name);
  }

  std::cout << "Family mtx file name: " << matrix_pathname << ", total number of matrices: " << num_systems << "\n";
  std::cout << "Family rhs file name: " << rhs_pathname    << ", total number of RHSes: "    << num_systems << "\n";

  // Workspace
  workspace_type workspace;
  workspace.initializeHandles();

  // Example helper
  ExampleHelper<workspace_type> helper(workspace);

  // Matrix and vector handlers
  MatrixHandler matrix_handler(&workspace);
  VectorHandler vector_handler(&workspace);

  // Direct solvers instantiation  
  LinSolverDirectKLU KLU;
  LinSolverDirectRocSolverRf Rf(&workspace);

  // Iterative solver instantiation
  GramSchmidt GS(&vector_handler, GramSchmidt::CGS2);
  LinSolverIterativeFGMRES FGMRES(&matrix_handler, &vector_handler, &GS);

  // Linear system matrix and vectors
  matrix::Csr* A = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  RESOLVE_RANGE_PUSH(__FUNCTION__);
  for (int i = 0; i < num_systems; ++i)
  {
    RESOLVE_RANGE_PUSH("Matrix Read");

    std::ostringstream matname;
    std::ostringstream rhsname;
    matname << matrix_pathname << std::setfill('0') << std::setw(2) << i << ".mtx";
    rhsname << rhs_pathname    << std::setfill('0') << std::setw(2) << i << ".mtx";
    std::string matrix_pathname_full = matname.str();
    std::string rhs_pathname_full    = rhsname.str();

    // Read matrix first
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << "Reading: " << matrix_pathname_full << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << std::endl;

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

    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x "<< A->getNumColumns() 
              << ", nnz: "       << A->getNnz() 
              << ", symmetric? " << A->symmetric()
              << ", Expanded? "  << A->expanded() << std::endl;

    RESOLVE_RANGE_POP("Matrix Read");
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    int status = 0;

    if (i < 2) {
      RESOLVE_RANGE_PUSH("KLU");
      KLU.setup(A);
      matrix_handler.setValuesChanged(true, memory::DEVICE);
      status = KLU.analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = KLU.factorize();
      std::cout << "KLU factorization status: " << status << std::endl;

      status = KLU.solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;

      helper.resetSystem(A, vec_rhs, vec_x);
      helper.printShortSummary();

      if (i == 1) {
        matrix::Csc* L = (matrix::Csc*) KLU.getLFactor();
        matrix::Csc* U = (matrix::Csc*) KLU.getUFactor();
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        index_type* P = KLU.getPOrdering();
        index_type* Q = KLU.getQOrdering();
        Rf.setSolveMode(1);
        Rf.setup(A, L, U, P, Q, vec_rhs);
        Rf.refactorize();
        std::cout << "About to set FGMRES ..." << std::endl;
        FGMRES.setup(A); 
      }
      RESOLVE_RANGE_POP("KLU");
    } else {
      RESOLVE_RANGE_PUSH("RocSolver");
      //status =  KLU.refactorize();
      std::cout << "Using ROCSOLVER RF" << std::endl;
      status = Rf.refactorize();
      std::cout << "ROCSOLVER RF refactorization status: " << status << std::endl;      
      status = Rf.solve(vec_rhs, vec_x);
      std::cout << "ROCSOLVER RF solve status: " << status << std::endl;      

      helper.resetSystem(A, vec_rhs, vec_x);

      //matrix_handler->setValuesChanged(true, memory::DEVICE);
      FGMRES.resetMatrix(A);
      FGMRES.setupPreconditioner("LU", &Rf);

      helper.printSummary();

      if (std::isfinite(helper.getNormRelativeResidual())) {
        FGMRES.solve(vec_rhs, vec_x);

        helper.printIrSummary(&FGMRES);
      }
      RESOLVE_RANGE_POP("RocSolver");
    }

  } // for (int i = 0; i < num_systems; ++i)
  RESOLVE_RANGE_POP(__FUNCTION__);

  delete A;
  delete vec_x;
  delete vec_rhs;

  return 0;
}
