#include <string>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

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

  ReSolve::matrix::Coo* A_coo;
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
  real_type norm_A, norm_x, norm_r;//used for INF norm
  
  ReSolve::GramSchmidt* GS = new ReSolve::GramSchmidt(vector_handler, ReSolve::GramSchmidt::cgs2);
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf = new ReSolve::LinSolverDirectCuSolverRf;
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler, GS);

  struct timeval t1;
  struct timeval t2;
  // struct timeval t3;
  // struct timeval t4;

  for (int i = 0; i < numSystems; ++i)
  {
    double time_io      = 0.0;
    double time_convert = 0.0;
    double time_factorize = 0.0;
    double time_solve   = 0.0;

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
    gettimeofday(&t1, 0);
    if (i == 0) {
      A_coo = ReSolve::io::readMatrixFromFile(mat_file);
      A = new ReSolve::matrix::Csr(A_coo->getNumRows(),
                                   A_coo->getNumColumns(),
                                   A_coo->getNnz(),
                                   A_coo->symmetric(),
                                   A_coo->expanded());

      rhs = ReSolve::io::readRhsFromFile(rhs_file);
      x = new real_type[A->getNumRows()];
      vec_rhs = new vector_type(A->getNumRows());
      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);//for KLU
      vec_x->allocate(ReSolve::memory::DEVICE);
      vec_r = new vector_type(A->getNumRows());
    }
    else {
      ReSolve::io::readAndUpdateMatrix(mat_file, A_coo);
      ReSolve::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    gettimeofday(&t2, 0);
    time_io += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    std::cout << "Finished reading the matrix and rhs, size: "
              << A->getNumRows() << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz() << ", symmetric? " << A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    gettimeofday(&t1, 0);
    if (i < 2) { 
      A->updateFromCoo(A_coo, ReSolve::memory::HOST);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else { 
      A->updateFromCoo(A_coo, ReSolve::memory::DEVICE);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    gettimeofday(&t2, 0);
    time_convert += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    std::cout << "COO to CSR completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;
    //Now call direct solver
    int status;
    real_type norm_b;
    if (i < 2) {
      // setup + factorize
      gettimeofday(&t1, 0);
      KLU->setup(A);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      status = KLU->analyze();
      // std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      gettimeofday(&t2, 0);
      time_factorize += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"KLU factorization status: " << status << std::endl;

      // solve
      gettimeofday(&t1, 0);
      status = KLU->solve(vec_rhs, vec_x);
      gettimeofday(&t2, 0);
      time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"KLU solve status: "<<status<<std::endl;

      // compute residual norms
      vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);
      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE,"csr", ReSolve::memory::DEVICE); 
    
      matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
      norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
      norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
      std::cout << "\t Matrix inf  norm: " << std::scientific << std::setprecision(16) << norm_A<<"\n"
                << "\t Residual inf norm: " << norm_r <<"\n"  
                << "\t Solution inf norm: " << norm_x <<"\n"  
                << "\t Norm of scaled residuals: "<< norm_r / (norm_A * norm_x) << "\n";
      
      std::cout << "\t2-Norm of the residual: "
                << std::scientific << std::setprecision(16) 
                << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/norm_b << "\n";
      if (i == 1) {
        gettimeofday(&t1, 0);
        ReSolve::matrix::Csc* L_csc = (ReSolve::matrix::Csc*) KLU->getLFactor();
        ReSolve::matrix::Csc* U_csc = (ReSolve::matrix::Csc*) KLU->getUFactor();
        ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
        matrix_handler->csc2csr(L_csc,L, ReSolve::memory::DEVICE);
        matrix_handler->csc2csr(U_csc,U, ReSolve::memory::DEVICE);
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        index_type* P = KLU->getPOrdering();
        index_type* Q = KLU->getQOrdering();
        Rf->setup(A, L, U, P, Q);
        // std::cout<<"about to set FGMRES" <<std::endl;
        GS->setup(A->getNumRows(), FGMRES->getRestart()); 
        FGMRES->setup(A); 
        gettimeofday(&t2, 0);
        time_factorize += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      }
    } else {
      //status =  KLU->refactorize();
      std::cout<<"Using CUSOLVER RF"<<std::endl;
      gettimeofday(&t1, 0);
      status = Rf->refactorize();
      gettimeofday(&t2, 0);
      time_factorize += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"CUSOLVER RF refactorization status: "<<status<<std::endl;      
      gettimeofday(&t1, 0);
      status = Rf->solve(vec_rhs, vec_x);
      gettimeofday(&t2, 0);
      time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"CUSOLVER RF solve status: "<<status<<std::endl;      

      vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      norm_b = vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE);
      norm_b = sqrt(norm_b);

      //matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
      gettimeofday(&t1, 0);
      FGMRES->resetMatrix(A);
      FGMRES->setupPreconditioner("LU", Rf);
      gettimeofday(&t2, 0);
      time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      
      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE,"csr", ReSolve::memory::DEVICE); 

      matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
      norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
      norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
      std::cout << "\t Matrix inf  norm: " << std::scientific << std::setprecision(16) << norm_A<<"\n"
        << "\t Residual inf norm: " << norm_r <<"\n"  
        << "\t Solution inf norm: " << norm_x <<"\n"  
        << "\t Norm of scaled residuals: "<< norm_r / (norm_A * norm_x) << "\n";
      
      std::cout << "\t 2-Norm of the residual (before IR): " 
                << std::scientific << std::setprecision(16) 
                << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/norm_b << "\n";

      matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      
      if(!std::isnan(norm_r) && !std::isinf(norm_r)) {
        gettimeofday(&t1, 0);
        FGMRES->solve(vec_rhs, vec_x);
        gettimeofday(&t2, 0);
        time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;

        std::cout << "FGMRES: init nrm: " 
          << std::scientific << std::setprecision(16) 
          << FGMRES->getInitResidualNorm()/norm_b
          << " final nrm: "
          << FGMRES->getFinalResidualNorm()/norm_b
          << " iter: " << FGMRES->getNumIter() << "\n";
      }
    }
    std::cout << std::setprecision(4) 
              << "I/O time: " << time_io << ", conversion time: " << time_convert
              << ", factorization time: " << time_factorize << ", solve time: " << time_solve
              << ", TOTAL: " << time_factorize + time_solve << "\n";
  } // for (int i = 0; i < numSystems; ++i)

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
