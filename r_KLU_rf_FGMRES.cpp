#include "MatrixIO.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "MatrixHandler.hpp"
#include "VectorHandler.hpp"
#include "LinSolverDirectKLU.hpp"
#include "LinSolverDirectCuSolverRf.hpp"
#include "LinSolverIterativeFGMRES.hpp"
#include <string>
#include <iostream>

int main(Int argc, char *argv[] ){

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  Int numSystems = atoi(argv[3]);
  std::cout<<"Family mtx file name: "<< matrixFileName << ", total number of matrices: "<<numSystems<<std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << ", total number of RHSes: " << numSystems<<std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::MatrixIO* reader = new ReSolve::MatrixIO;
  ReSolve::Matrix* A;
  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  Real* rhs;
  Real* x;

  ReSolve::Vector* vec_rhs;
  ReSolve::Vector* vec_x;
  ReSolve::Vector* vec_r;

  Real one = 1.0;
  Real minusone = -1.0;

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf = new ReSolve::LinSolverDirectCuSolverRf;
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler);

  for (int i = 0; i < numSystems; ++i)
  {
    Int j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout<<std::endl<<std::endl<<std::endl;
    std::cout<<"========================================================================================================================"<<std::endl;
    std::cout<<"Reading: "<<matrixFileNameFull<<std::endl;
    std::cout<<"========================================================================================================================"<<std::endl;
    std::cout<<std::endl;
    if (i == 0) {
      A = reader->readMatrixFromFile(matrixFileNameFull);

      rhs = reader->readRhsFromFile(rhsFileNameFull);
      x = new Real[A->getNumRows()];
      vec_rhs = new ReSolve::Vector(A->getNumRows());
      vec_x = new ReSolve::Vector(A->getNumRows());
      vec_x->allocate("cpu");//for KLU
      vec_x->allocate("cuda");
      vec_r = new ReSolve::Vector(A->getNumRows());
    }
    else {
      reader->readAndUpdateMatrix(matrixFileNameFull, A);
      reader->readAndUpdateRhs(rhsFileNameFull, rhs);
    }
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;

    //Now convert to CSR.
    if (i < 2) { 
      matrix_handler->coo2csr(A, "cpu");
      vec_rhs->update(rhs, "cpu", "cpu");
      vec_rhs->setDataUpdated("cpu");
    } else { 
      matrix_handler->coo2csr(A, "cuda");
      vec_rhs->update(rhs, "cpu", "cuda");
    }
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnzExpanded()<<std::endl;
    //Now call direct solver
    if (i == 0) {
      KLU->setupParameters(1, 0.1, false);
    }
    int status;
    Real norm_b;
    if (i < 2){
      KLU->setup(A);
      matrix_handler->setValuesChanged(true);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
      vec_r->update(rhs, "cpu", "cuda");
      norm_b = vector_handler->dot(vec_r, vec_r, "cuda");
      norm_b = sqrt(norm_b);
      matrix_handler->setValuesChanged(true);
      matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "cuda"); 
      printf("\t 2-Norm of the residual : %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda"))/norm_b);
      if (i == 1) {
        ReSolve::Matrix* L = KLU->getLFactor();
        ReSolve::Matrix* U = KLU->getUFactor();
        matrix_handler->csc2csr(L, "cuda");
        matrix_handler->csc2csr(U, "cuda");
        if (L == nullptr) {printf("ERROR");}
        Int* P = KLU->getPOrdering();
        Int* Q = KLU->getQOrdering();
        Rf->setup(A, L, U, P, Q);
        std::cout<<"about to set FGMRES" <<std::endl;
        FGMRES->setup(A); 
      }
    } else {
      //status =  KLU->refactorize();
      std::cout<<"Using CUSOLVER RF"<<std::endl;
      status = Rf->refactorize();
      std::cout<<"CUSOLVER RF refactorization status: "<<status<<std::endl;      
      status = Rf->solve(vec_rhs, vec_x);
      std::cout<<"CUSOLVER RF solve status: "<<status<<std::endl;      

      vec_r->update(rhs, "cpu", "cuda");
       norm_b = vector_handler->dot(vec_r, vec_r, "cuda");
      norm_b = sqrt(norm_b);

      //matrix_handler->setValuesChanged(true);
      FGMRES->resetMatrix(A);
      FGMRES->setupPreconditioner("CuSolverRf", Rf);
      
      matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone, "cuda"); 

      printf("\t 2-Norm of the residual (before IR): %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda"))/norm_b);

      vec_rhs->update(rhs, "cpu", "cuda");
      FGMRES->solve(vec_rhs, vec_x);

      printf("FGMRES: init nrm: %16.16e final nrm: %16.16e iter: %d \n", FGMRES->getInitResidualNorm()/norm_b, FGMRES->getFinalResidualNorm()/norm_b, FGMRES->getNumIter());
    }


  }

  return 0;
}
