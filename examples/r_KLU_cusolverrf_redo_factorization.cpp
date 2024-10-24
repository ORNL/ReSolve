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
#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;

int main(int argc, char *argv[] )
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

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf = new ReSolve::LinSolverDirectCuSolverRf();

  real_type res_nrm;
  real_type b_nrm;

  // We need them. They hold a POINTER. Don't delete them here. KLU deletes them.
  ReSolve::matrix::Csc* L_csc;
  ReSolve::matrix::Csc* U_csc;
  index_type* P;
  index_type* Q;

  int status;
  int status_refactor = 0;
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

    // Update host and device data.
    if (i < 2) { 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    } else { 
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    //Now call direct solver
    if (i < 2) {
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      
      if (i == 1) {
        L_csc = (ReSolve::matrix::Csc*) KLU->getLFactor();
        U_csc = (ReSolve::matrix::Csc*) KLU->getUFactor();
        ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
        ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
        L_csc->syncData(ReSolve::memory::DEVICE);
        U_csc->syncData(ReSolve::memory::DEVICE);
        matrix_handler->csc2csr(L_csc,L, ReSolve::memory::DEVICE);
        matrix_handler->csc2csr(U_csc,U, ReSolve::memory::DEVICE);
        if (L == nullptr) {
          std::cout << "ERROR\n";
        }
        P = KLU->getPOrdering();
        Q = KLU->getQOrdering();
        Rf->setup(A, L, U, P, Q); 
        Rf->refactorize();
        delete L;
        delete U;
      }
    } else {
      std::cout<<"Using cusolver rf"<<std::endl;
      status_refactor = Rf->refactorize();
      std::cout<<"cusolver rf refactorization status: "<<status<<std::endl;      
      status = Rf->solve(vec_rhs, vec_x);
      std::cout<<"cusolver rf solve status: "<<status<<std::endl;      
    }
    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

    matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
    res_nrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
    b_nrm   = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));
    std::cout << "\t2-Norm of the residual: " 
              << std::scientific << std::setprecision(16) 
              << res_nrm/b_nrm << "\n";
    if (((res_nrm/b_nrm > 1e-7 ) && (!std::isnan(res_nrm))) || (status_refactor != 0 )) {
      if ((res_nrm/b_nrm > 1e-7 )) {
        std::cout << "\n \t !!! ALERT !!! Residual norm is too large; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";
      } else { 
        std::cout << "\n \t !!! ALERT !!! cuSolverRf crashed; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";
      }
      KLU->setup(A);
      status = KLU->analyze();
      std::cout<<"KLU analysis status: "<<status<<std::endl;
      status = KLU->factorize();
      std::cout<<"KLU factorization status: "<<status<<std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout<<"KLU solve status: "<<status<<std::endl;      

      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, ReSolve::memory::DEVICE); 
      res_nrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));

      std::cout <<"\t New residual norm: "
                << std::scientific << std::setprecision(16)
                << res_nrm/b_nrm << "\n";


      L_csc = (ReSolve::matrix::Csc*) KLU->getLFactor();
      U_csc = (ReSolve::matrix::Csc*) KLU->getUFactor();

      ReSolve::matrix::Csr* L = new ReSolve::matrix::Csr(L_csc->getNumRows(), L_csc->getNumColumns(), L_csc->getNnz());
      ReSolve::matrix::Csr* U = new ReSolve::matrix::Csr(U_csc->getNumRows(), U_csc->getNumColumns(), U_csc->getNnz());
      matrix_handler->csc2csr(L_csc, L, ReSolve::memory::DEVICE);
      matrix_handler->csc2csr(U_csc, U, ReSolve::memory::DEVICE);

      P = KLU->getPOrdering();
      Q = KLU->getQOrdering();

      Rf->setup(A, L, U, P, Q); 
      Rf->refactorize();

      delete L;
      delete U;
    }
  } // for (int i = 0; i < numSystems; ++i)

  //now DELETE
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
