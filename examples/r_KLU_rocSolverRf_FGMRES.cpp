#include <string>
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
  // Use the same data types as those you specified in ReSolve build.
  using namespace ReSolve::constants;
  using namespace ReSolve::examples;
  using namespace ReSolve;
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
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
  workspace_type* workspace_HIP = new ReSolve::LinAlgWorkspaceHIP();
  workspace_HIP->initializeHandles();

  // Example helper
  ExampleHelper<workspace_type> helper(*workspace_HIP);

  // Matrix and vector handlers
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_HIP);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_HIP);

  // Direct solvers instantiation  
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectRocSolverRf* Rf = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP);

  // Iterative solver instantiation
  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::CGS2);
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);

  // Linear system matrix and vectors
  ReSolve::matrix::Csr* A = nullptr;
  real_type* rhs = nullptr;
  real_type* x   = nullptr;
  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  // Auxilliary objects for error estimates
  vector_type* vec_r   = nullptr;
  real_type norm_A{0.0};
  real_type norm_x{0.0};
  real_type norm_r{0.0};

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
    // Read first matrix
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
      // vec_rhs = io::createVectorFromFile(rhs_file);
      x = new real_type[A->getNumRows()];
      vec_rhs = new vector_type(A->getNumRows());
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);//for KLU
      vec_x->allocate(ReSolve::memory::DEVICE);
      vec_r = new vector_type(A->getNumRows());
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
      // io::updateVectorFromFile(rhs_file, vec_rhs);
    }
    // Copy matrix data to device
    A->syncData(ReSolve::memory::DEVICE);

    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x "<< A->getNumColumns() 
              << ", nnz: "       << A->getNnz() 
              << ", symmetric? " << A->symmetric()
              << ", Expanded? "  << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    // Update host and device data.
    if (i < 2) { 
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    } else { 
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      // vec_rhs->syncData(memory::DEVICE);
    }
    RESOLVE_RANGE_POP("Matrix Read");
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    int status;
    real_type norm_b;
    if (i < 2) {
      RESOLVE_RANGE_PUSH("KLU");
      KLU->setup(A);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      status = KLU->analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = KLU->factorize();
      std::cout << "KLU factorization status: " << status << std::endl;

      status = KLU->solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;      
      vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
      std::cout << "\t2-Norm of the residual: "
        << std::scientific << std::setprecision(16) 
        << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/norm_b << "\n";
      if (i == 1) {
        ReSolve::matrix::Csc* L = (ReSolve::matrix::Csc*) KLU->getLFactor();
        ReSolve::matrix::Csc* U = (ReSolve::matrix::Csc*) KLU->getUFactor();
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        index_type* P = KLU->getPOrdering();
        index_type* Q = KLU->getQOrdering();
        Rf->setSolveMode(1);
        Rf->setup(A, L, U, P, Q, vec_rhs);
        Rf->refactorize();
        std::cout << "About to set FGMRES ..." << std::endl;
        FGMRES->setup(A); 
      }
      RESOLVE_RANGE_POP("KLU");
    } else {
      RESOLVE_RANGE_PUSH("RocSolver");
      //status =  KLU->refactorize();
      std::cout << "Using ROCSOLVER RF" << std::endl;
      status = Rf->refactorize();
      std::cout << "ROCSOLVER RF refactorization status: " << status << std::endl;      
      status = Rf->solve(vec_rhs, vec_x);

      helper.resetSystem(A, vec_rhs, vec_x);

      std::cout << "ROCSOLVER RF solve status: " << status << std::endl;      
      vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);

      //matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      FGMRES->resetMatrix(A);
      FGMRES->setupPreconditioner("LU", Rf);

      helper.printSummary();

      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
      real_type rnrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
      // std::cout << "\t 2-Norm of the residual (before IR): " 
      //   << std::scientific << std::setprecision(16) 
      //   << rnrm/norm_b << "\n";

      matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
      norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
      norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
      // std::cout << std::scientific << std::setprecision(16)
      //           << "\t Matrix inf  norm: "         << norm_A << "\n"
      //           << "\t Residual inf norm: "        << norm_r << "\n"  
      //           << "\t Solution inf norm: "        << norm_x << "\n"  
      //           << "\t Norm of scaled residuals: " << norm_r / (norm_A * norm_x) << "\n";

      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      // vec_rhs->syncData(memory::DEVICE);
      if(!std::isnan(rnrm) && !std::isinf(rnrm)) {
        FGMRES->solve(vec_rhs, vec_x);

        helper.printIrSummary(FGMRES);

        // std::cout << "FGMRES: init nrm: " 
        //           << std::scientific << std::setprecision(16) 
        //           << FGMRES->getInitResidualNorm()/norm_b
        //           << " final nrm: "
        //           << FGMRES->getFinalResidualNorm()/norm_b
        //           << " iter: " << FGMRES->getNumIter() << "\n";
      }
      RESOLVE_RANGE_POP("RocSolver");
    }

  } // for (int i = 0; i < num_systems; ++i)
  RESOLVE_RANGE_POP(__FUNCTION__);

  delete A;
  delete KLU;
  delete Rf;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete workspace_HIP;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
