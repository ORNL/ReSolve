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
  std::cout << "Family mtx file name: " << matrixFileName << ", total number of matrices: " << numSystems << std::endl;
  std::cout << "Family rhs file name: " << rhsFileName    << ", total number of RHSes: "    << numSystems << std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::matrix::Csr* A;
  ReSolve::LinAlgWorkspaceHIP* workspace_HIP = new ReSolve::LinAlgWorkspaceHIP();
  workspace_HIP->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_HIP);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_HIP);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;
  real_type norm_A;
  real_type norm_x;
  real_type norm_r;

  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectRocSolverRf* Rf = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP);
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);

  RESOLVE_RANGE_PUSH(__FUNCTION__);
  for (int i = 0; i < numSystems; ++i)
  {
    RESOLVE_RANGE_PUSH("Matrix Read");
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
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);//for KLU
      vec_x->allocate(ReSolve::memory::DEVICE);
      vec_r = new vector_type(A->getNumRows());
    } else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
    }
    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x "<< A->getNumColumns() 
              << ", nnz: "       << A->getNnz() 
              << ", symmetric? " << A->symmetric()
              << ", Expanded? "  << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();
    RESOLVE_RANGE_POP("Matrix Read");

    // Update host and device data.
    RESOLVE_RANGE_PUSH("Convert to CSR");
    if (i < 2) { 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else { 
      A->copyData(ReSolve::memory::DEVICE);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    RESOLVE_RANGE_POP("Convert to CSR");
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnz()<<std::endl;
    int status;
    real_type norm_b;
    if (i < 2) {
      RESOLVE_RANGE_PUSH("KLU");
      KLU->setup(A);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;

      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
      vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
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
        std::cout<<"about to set FGMRES" <<std::endl;
        FGMRES->setup(A); 
      }
      RESOLVE_RANGE_POP("KLU");
    } else {
      RESOLVE_RANGE_PUSH("RocSolver");
      //status =  KLU->refactorize();
      std::cout<<"Using ROCSOLVER RF"<<std::endl;
      status = Rf->refactorize();
      std::cout<<"ROCSOLVER RF refactorization status: "<<status<<std::endl;      
      status = Rf->solve(vec_rhs, vec_x);
      std::cout<<"ROCSOLVER RF solve status: "<<status<<std::endl;      
      vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);

      //matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      FGMRES->resetMatrix(A);
      FGMRES->setupPreconditioner("LU", Rf);

      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
      real_type rnrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
      std::cout << "\t 2-Norm of the residual (before IR): " 
        << std::scientific << std::setprecision(16) 
        << rnrm/norm_b << "\n";

      matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
      norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
      norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
      std::cout << std::scientific << std::setprecision(16)
                << "\t Matrix inf  norm: "         << norm_A << "\n"
                << "\t Residual inf norm: "        << norm_r << "\n"  
                << "\t Solution inf norm: "        << norm_x << "\n"  
                << "\t Norm of scaled residuals: " << norm_r / (norm_A * norm_x) << "\n";

      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      if(!std::isnan(rnrm) && !std::isinf(rnrm)) {
        FGMRES->solve(vec_rhs, vec_x);

        std::cout << "FGMRES: init nrm: " 
                  << std::scientific << std::setprecision(16) 
                  << FGMRES->getInitResidualNorm()/norm_b
                  << " final nrm: "
                  << FGMRES->getFinalResidualNorm()/norm_b
                  << " iter: " << FGMRES->getNumIter() << "\n";
      }
      RESOLVE_RANGE_POP("RocSolver");
    }

  } // for (int i = 0; i < numSystems; ++i)
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
