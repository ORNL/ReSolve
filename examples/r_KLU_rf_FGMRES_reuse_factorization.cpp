#include <string>
#include <iostream>
#include <iomanip>

#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

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

  ReSolve::matrix::Csr* A;

  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;

  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::CGS2);
  
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf = new ReSolve::LinSolverDirectCuSolverRf;
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);

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
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);//for KLU
      vec_x->allocate(ReSolve::memory::DEVICE);
      vec_r = new vector_type(A->getNumRows());
    }
    else {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
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
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else { 
      A->syncData(ReSolve::memory::DEVICE);
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    //Now call direct solver
    int status;
    real_type norm_b;
    if (i < 2){
      KLU->setup(A);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
      vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
      std::cout << "\t 2-Norm of the residual : " 
                << std::scientific << std::setprecision(16) 
                << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/norm_b << "\n";
      if (i == 1) {
        ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*) KLU->getLFactor();
        ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*) KLU->getUFactor();
        ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
        L_csc->syncData(ReSolve::memory::DEVICE);
        U_csc->syncData(ReSolve::memory::DEVICE);

        matrix_handler->csc2csr(L_csc,L, ReSolve::memory::DEVICE);
        matrix_handler->csc2csr(U_csc,U, ReSolve::memory::DEVICE);
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        index_type* P = KLU->getPOrdering();
        index_type* Q = KLU->getQOrdering();
        Rf->setup(A, L, U, P, Q);
        std::cout<<"about to set FGMRES" <<std::endl;
        FGMRES->setRestart(1000); 
        FGMRES->setMaxit(2000);
        FGMRES->setup(A);
      }
    } else {
      //status =  KLU->refactorize();
      std::cout<<"Using CUSOLVER RF"<<std::endl;
      if ((i % 2 == 0))
      {
        status = Rf->refactorize();
        std::cout << "CUSOLVER RF, using REAL refactorization, refactorization status: "
                  << status << std::endl;    
        vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        status = Rf->solve(vec_rhs, vec_x);
        FGMRES->setupPreconditioner("LU", Rf);
      }
      //if (i%2!=0)  vec_x->setToZero(ReSolve::memory::DEVICE);
      real_type norm_x =  vector_handler->dot(vec_x, vec_x, ReSolve::memory::DEVICE);
      std::cout << "Norm of x (before solve): " 
                << std::scientific << std::setprecision(16) 
                << sqrt(norm_x) << "\n";
      std::cout<<"CUSOLVER RF solve status: "<<status<<std::endl;      
      
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);

      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      FGMRES->resetMatrix(A);
      
      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 

      std::cout << "\t 2-Norm of the residual (before IR): " 
                << std::scientific << std::setprecision(16) 
                << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/norm_b << "\n";
      std::cout << "\t 2-Norm of the RIGHT HAND SIDE: " 
                << std::scientific << std::setprecision(16) 
                << norm_b << "\n";

      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      FGMRES->solve(vec_rhs, vec_x);

      std::cout << "FGMRES: init nrm: " 
                << std::scientific << std::setprecision(16) 
                << FGMRES->getInitResidualNorm()/norm_b
                << " final nrm: "
                << FGMRES->getFinalResidualNorm()/norm_b
                << " iter: " << FGMRES->getNumIter() << "\n";
      norm_x = vector_handler->dot(vec_x, vec_x, ReSolve::memory::DEVICE);
      std::cout << "Norm of x (after IR): " 
                << std::scientific << std::setprecision(16) 
                << sqrt(norm_x) << "\n";
    }


  }

  delete A;
  delete KLU;
  delete Rf;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete workspace_CUDA;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
