#include <string>
#include <iostream>

#include <resolve/MatrixCOO.hpp>
#include <resolve/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/MatrixHandler.hpp>
#include <resolve/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  std::string  matrixFileName = argv[1];
  std::string  rhsFileName = argv[2];

  index_type numSystems = atoi(argv[3]);
  std::cout<<"Family mtx file name: "<< matrixFileName << ", total number of matrices: "<<numSystems<<std::endl;
  std::cout<<"Family rhs file name: "<< rhsFileName << ", total number of RHSes: " << numSystems<<std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::MatrixCOO* A_coo;
  ReSolve::MatrixCSR* A;

  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_CUDA);
  real_type* rhs;
  real_type* x;

  ReSolve::Vector* vec_rhs;
  ReSolve::Vector* vec_x;
  ReSolve::Vector* vec_r;

  real_type one = 1.0;
  real_type minusone = -1.0;

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf = new ReSolve::LinSolverDirectCuSolverRf;
  ReSolve::LinSolverIterativeFGMRES* FGMRES = new ReSolve::LinSolverIterativeFGMRES(matrix_handler, vector_handler);

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
    if (i == 0) {
      A_coo = ReSolve::matrix::io::readMatrixFromFile(mat_file);
      A = new ReSolve::MatrixCSR(A_coo->getNumRows(), A_coo->getNumColumns(), A_coo->getNnz(), A_coo->expanded(), A_coo->symmetric());
      rhs = ReSolve::matrix::io::readRhsFromFile(rhs_file);
      x = new real_type[A->getNumRows()];
      vec_rhs = new ReSolve::Vector(A->getNumRows());
      vec_x = new ReSolve::Vector(A->getNumRows());
      vec_x->allocate("cpu");//for KLU
      vec_x->allocate("cuda");
      vec_r = new ReSolve::Vector(A->getNumRows());
    }
    else {
      ReSolve::matrix::io::readAndUpdateMatrix(mat_file, A_coo);
      ReSolve::matrix::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    if (i < 2) { 
      matrix_handler->coo2csr(A_coo,A, "cpu");
      vec_rhs->update(rhs, "cpu", "cpu");
      vec_rhs->setDataUpdated("cpu");
    } else { 
      matrix_handler->coo2csr(A_coo, A, "cuda");
      vec_rhs->update(rhs, "cpu", "cuda");
    }
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnzExpanded()<<std::endl;
    //Now call direct solver
    if (i == 0) {
      KLU->setupParameters(1, 0.1, false);
    }
    int status;
    real_type norm_b;
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
      matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone,"csr", "cuda"); 
      printf("\t 2-Norm of the residual : %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda"))/norm_b);
      if (i == 1) {
        ReSolve::MatrixCSC* L_csc = (ReSolve::MatrixCSC*) KLU->getLFactor();
        ReSolve::MatrixCSC* U_csc = (ReSolve::MatrixCSC*) KLU->getUFactor();
        ReSolve::MatrixCSR* L = new ReSolve::MatrixCSR(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::MatrixCSR* U = new ReSolve::MatrixCSR(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
;
        matrix_handler->csc2csr(L_csc,L, "cuda");
        matrix_handler->csc2csr(U_csc,U, "cuda");
        if (L == nullptr) {printf("ERROR");}
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
        std::cout<<"CUSOLVER RF, using REAL refactorization, refactorization status: "<<status<<std::endl;      
        vec_rhs->update(rhs, "cpu", "cuda");
        status = Rf->solve(vec_rhs, vec_x);
        FGMRES->setupPreconditioner("CuSolverRf", Rf);
      }
     //if (i%2!=0)  vec_x->setToZero("cuda");
      real_type norm_x =  vector_handler->dot(vec_x, vec_x, "cuda");
      printf("Norm of x(before solve): %16.16e \n", sqrt(norm_x));
      std::cout<<"CUSOLVER RF solve status: "<<status<<std::endl;      
      
      vec_rhs->update(rhs, "cpu", "cuda");
      vec_r->update(rhs, "cpu", "cuda");
       norm_b = vector_handler->dot(vec_r, vec_r, "cuda");
      norm_b = sqrt(norm_b);

      matrix_handler->setValuesChanged(true);
      FGMRES->resetMatrix(A);
      
      matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone,"csr", "cuda"); 

      printf("\t 2-Norm of the residual (before IR): %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda"))/norm_b);
      printf("\t 2-Norm of the RIGHT HAND SIDE: %16.16e\n", norm_b);

      vec_rhs->update(rhs, "cpu", "cuda");
      FGMRES->solve(vec_rhs, vec_x);

      printf("FGMRES: init nrm: %16.16e final nrm: %16.16e iter: %d \n", FGMRES->getInitResidualNorm()/norm_b, FGMRES->getFinalResidualNorm()/norm_b, FGMRES->getNumIter());
      norm_x = vector_handler->dot(vec_x, vec_x, "cuda");
      printf("Norm of x (after IR): %16.16e \n", sqrt(norm_x));
    }


  }

  return 0;
}
