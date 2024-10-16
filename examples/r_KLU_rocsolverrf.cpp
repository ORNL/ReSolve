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
  std::cout << "Family mtx file name: " << matrixFileName << ", total number of matrices: " << numSystems << std::endl;
  std::cout << "Family rhs file name: " << rhsFileName << ", total number of RHSes: " << numSystems << std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::matrix::Csr* A;

  ReSolve::LinAlgWorkspaceHIP* workspace_HIP = new ReSolve::LinAlgWorkspaceHIP;
  workspace_HIP->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler =  new ReSolve::MatrixHandler(workspace_HIP);
  ReSolve::VectorHandler* vector_handler =  new ReSolve::VectorHandler(workspace_HIP);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;

  real_type norm_A, norm_x, norm_r; //used for INF norm
  
  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectRocSolverRf* Rf = new ReSolve::LinSolverDirectRocSolverRf(workspace_HIP);

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
      vec_r = new vector_type(A->getNumRows());
    } else {
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

    // Update right-hand-side vector.
    if (i < 2) { 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    } else { 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    // Now call direct solver
    int status;
    if (i < 2) {
      KLU->setup(A);
      status = KLU->analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = KLU->factorize();
      std::cout << "KLU factorization status: " << status << std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;      
      if (i == 1) {
        ReSolve::matrix::Csc* L = (ReSolve::matrix::Csc*) KLU->getLFactor();
        ReSolve::matrix::Csc* U = (ReSolve::matrix::Csc*) KLU->getUFactor();
        index_type* P = KLU->getPOrdering();
        index_type* Q = KLU->getQOrdering();
        vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
        Rf->setup(A, L, U, P, Q, vec_rhs); 
      }
    } else {
      std::cout << "Using rocsolver rf" << std::endl;
      status = Rf->refactorize();
      std::cout << "rocsolver rf refactorization status: " << status << std::endl;      
      status = Rf->solve(vec_rhs, vec_x);
      std::cout << "rocsolver rf solve status: " << status << std::endl;      
    }

    // Check accuracy of the solution
    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    real_type bnorm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
    matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);
    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 

    std::cout << "\t 2-Norm of the residual: " 
              << std::scientific << std::setprecision(16) 
              << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE))/bnorm << "\n";
    
    matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::DEVICE); 
    norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::DEVICE);
    norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::DEVICE);
    std::cout << "\t Matrix inf  norm:  " << std::scientific << std::setprecision(16) << norm_A <<"\n"
              << "\t Residual inf norm: " << norm_r <<"\n"  
              << "\t Solution inf norm: " << norm_x <<"\n"  
              << "\t Norm of scaled residuals: "<< norm_r / (norm_A * norm_x) << "\n";

  } // for (int i = 0; i < numSystems; ++i)

  // now DELETE
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
