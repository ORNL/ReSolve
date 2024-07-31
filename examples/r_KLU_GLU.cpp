#include <string>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/LinSolverDirectCuSolverGLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;

int main(int argc, char *argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;
  using matrix_type = ReSolve::matrix::Sparse;

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

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverGLU* GLU = new ReSolve::LinSolverDirectCuSolverGLU(workspace_CUDA);

  struct timeval t1;
  struct timeval t2;
    
  double time_io      = 0.0;
  double time_convert = 0.0;
  double time_factorize = 0.0;
  double time_solve   = 0.0;
  for (int i = 0; i < numSystems; ++i)
  {
    time_io      = 0.0;
    time_convert = 0.0;
    time_factorize = 0.0;
    time_solve   = 0.0;
   
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
    // Time system I/O
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
    } else {
      ReSolve::io::readAndUpdateMatrix(mat_file, A_coo);
      ReSolve::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    gettimeofday(&t2, 0);
    time_io += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    std::cout<<"Finished reading the matrix and rhs, size: "<<A->getNumRows()<<" x "<<A->getNumColumns()<< ", nnz: "<< A->getNnz()<< ", symmetric? "<<A->symmetric()<< ", Expanded? "<<A->expanded()<<std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    // Time matrix conversion
    gettimeofday(&t1, 0);
    if (i < 1) {
      A->updateFromCoo(A_coo, ReSolve::memory::HOST);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else { 
      A->updateFromCoo(A_coo, ReSolve::memory::DEVICE);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    gettimeofday(&t2, 0);
    time_convert += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnzExpanded()<<std::endl;
    //Now call direct solver
    int status;
    if (i < 1) {
      // Time factorization (CPU part)
      gettimeofday(&t1, 0);
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      gettimeofday(&t2, 0);
      matrix_type* L = KLU->getLFactor();
      matrix_type* U = KLU->getUFactor();
      if (L == nullptr) {printf("ERROR");}
      index_type* P = KLU->getPOrdering();
      index_type* Q = KLU->getQOrdering();
      GLU->setup(A, L, U, P, Q); 
      time_factorize += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      // time solve (cpu part)
      gettimeofday(&t1, 0);
      status = GLU->solve(vec_rhs, vec_x);
      gettimeofday(&t2, 0);
      time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"GLU solve status: "<<status<<std::endl;      
    } else {
      std::cout<<"Using CUSOLVER GLU"<<std::endl;
        gettimeofday(&t1, 0);
      status = GLU->refactorize();
        gettimeofday(&t2, 0);
        time_factorize += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"CUSOLVER GLU refactorization status: "<<status<<std::endl;      
      gettimeofday(&t1, 0);
      status = GLU->solve(vec_rhs, vec_x);
      gettimeofday(&t2, 0);
      time_solve += (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0;
      std::cout<<"CUSOLVER GLU solve status: "<<status<<std::endl;      
    }

    // Estimate solution error
    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    real_type bnorm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
    matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE,"csr", ReSolve::memory::DEVICE); 

    
    matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
    norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
    norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
    std::cout << "\t Matrix inf  norm: " << std::scientific << std::setprecision(16) << norm_A<<"\n"
      << "\t Residual inf norm: " << norm_r <<"\n"  
      << "\t Solution inf norm: " << norm_x <<"\n"  
      << "\t Norm of scaled residuals: "<< norm_r / (norm_A * norm_x) << "\n";
    
    std::cout << "\t 2-Norm of the residual: " 
              << std::scientific << std::setprecision(16) 
              << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/bnorm << "\n";

    // Print timing summary
    std::cout << std::defaultfloat << std::setprecision(4) 
              << "I/O time: " << time_io << ", conversion time: " << time_convert
              << ", factorization time: " << time_factorize << ", solve time: " << time_solve
              << "\nTOTAL: " << time_factorize + time_solve << "\n";
  } // for (int i = 0; i < numSystems; ++i)

  //now DELETE
  delete A;
  delete KLU;
  delete GLU;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete workspace_CUDA;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
