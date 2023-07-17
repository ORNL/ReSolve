#include <string>
#include <iostream>

#include <resolve/MatrixCOO.hpp>
#include <resolve/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/MatrixHandler.hpp>
#include <resolve/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>

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
  ReSolve::LinSolverDirectCuSolverGLU* GLU = new ReSolve::LinSolverDirectCuSolverGLU(workspace_CUDA);

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
    if (i < 1) { 
      matrix_handler->coo2csr(A_coo, A,  "cpu");
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
    if (i < 1){
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      if (i == 0) {
        ReSolve::Matrix* L = KLU->getLFactor();
        ReSolve::Matrix* U = KLU->getUFactor();
        if (L == nullptr) {printf("ERROR");}
        index_type* P = KLU->getPOrdering();
        index_type* Q = KLU->getQOrdering();
        GLU->setup(A, L, U, P, Q); 
        status = GLU->solve(vec_rhs, vec_x);
        std::cout<<"GLU solve status: "<<status<<std::endl;      
      } else { 
        status = KLU->solve(vec_rhs, vec_x);
        std::cout<<"KLU solve status: "<<status<<std::endl;      
      }
    } else {
      //status =  KLU->refactorize();
      std::cout<<"Using CUSOLVER GLU"<<std::endl;
      status = GLU->refactorize();
      std::cout<<"CUSOLVER GLU refactorization status: "<<status<<std::endl;      
      status = GLU->solve(vec_rhs, vec_x);
      std::cout<<"CUSOLVER GLU solve status: "<<status<<std::endl;      
    }
    vec_r->update(rhs, "cpu", "cuda");


    matrix_handler->setValuesChanged(true);
    matrix_handler->matvec(A, vec_x, vec_r, &one, &minusone,"csr", "cuda"); 

    printf("\t 2-Norm of the residual: %16.16e\n", sqrt(vector_handler->dot(vec_r, vec_r, "cuda")));


  }

  //now DELETE
  delete A;
  delete KLU;
  delete GLU;
  delete x;
  delete vec_r;
  delete vec_x;
  delete workspace_CUDA;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
