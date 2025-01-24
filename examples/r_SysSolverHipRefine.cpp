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
#include <resolve/SystemSolver.hpp>

using namespace ReSolve::constants;

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  (void) argc; // TODO: Check if the number of input parameters is correct.
  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  index_type numSystems = atoi(argv[3]);
  std::cout<<"Family mtx file name: "<< matrixFileName << ", total number of matrices: "<<numSystems<<std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << ", total number of RHSes: " << numSystems<<std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::matrix::Csr* A = nullptr;
  ReSolve::LinAlgWorkspaceHIP* workspace_HIP = new ReSolve::LinAlgWorkspaceHIP();
  workspace_HIP->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_HIP);

  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;

  ReSolve::SystemSolver* solver = new ReSolve::SystemSolver(workspace_HIP);
  solver->setRefinementMethod("fgmres", "CGS2");

  for (int i = 0; i < numSystems; ++i)
  {
    index_type j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << "Reading: " << matrixFileNameFull << std::endl;
    std::cout << "========================================================================================================================"<<std::endl;
    std::cout << std::endl;
    // Read first matrix
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
      vec_x =   new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);//for KLU
      vec_x->allocate(ReSolve::memory::DEVICE);
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
    }
    // Copy matrix data to device
    A->syncData(ReSolve::memory::DEVICE);

    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows()
              << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz()
              << ", symmetric? "<< A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    // Update host and device data.
    if (i < 2) { 
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    } else { 
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    // Now call solver
    solver->setMatrix(A);
    int status;
    if (i < 2) {
      status = solver->analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = solver->factorize();
      std::cout << "KLU factorization status: " << status << std::endl;
     
      status = solver->solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;      

      std::cout << "\t 2-Norm of the residual      : " 
                << std::scientific << std::setprecision(16)
                << solver->getResidualNorm(vec_rhs, vec_x) << "\n";
      if (i == 1) {
        solver->refactorizationSetup();
        std::cout << "rocsolver rf refactorization setup status: " << status << std::endl;
      }
    } else {
      std::cout << "Using ROCSOLVER RF" << std::endl;
      status = solver->refactorize();
      std::cout << "ROCSOLVER RF refactorization status: " << status << std::endl;      
      status = solver->solve(vec_rhs, vec_x);
      std::cout << "ROCSOLVER RF solve status: " << status << std::endl;

      real_type rnrm = solver->getResidualNorm(vec_rhs, vec_x);
      std::cout << std::scientific << std::setprecision(16) 
                << "\t 2-Norm of the residual (after IR): " 
                << rnrm 
                << "\t 2-Norm of scaled residuals (after IR): "
                << solver->getNormOfScaledResiduals(vec_rhs, vec_x)
                << "\n";

      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      real_type norm_b = solver->getVectorNorm(vec_rhs);
      if (!std::isnan(rnrm) && !std::isinf(rnrm)) {
        std::cout << "FGMRES solve status: " << status << std::endl;      

        std::cout << "FGMRES: init nrm: " 
                  << std::scientific << std::setprecision(16) 
                  << solver->getIterativeSolver().getInitResidualNorm()/norm_b
                  << " final nrm: "
                  << solver->getIterativeSolver().getFinalResidualNorm()/norm_b
                  << " iter: " << solver->getIterativeSolver().getNumIter()
                  << "\n";
      }
    }

  } // for (int i = 0; i < numSystems; ++i)

  delete A;
  delete solver;
  delete [] x;
  delete [] rhs;
  delete vec_rhs;
  delete vec_x;
  delete workspace_HIP;
  delete matrix_handler;

  return 0;
}
