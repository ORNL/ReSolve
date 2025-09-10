#include <iomanip>
#include <iostream>
#include <string>

#include <resolve/LinSolverDirectCuSolverRf.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;

int main(int argc, char* argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type  = ReSolve::index_type;
  using real_type   = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  (void) argc; // TODO: Check if the number of input parameters is correct.
  std::string matrixFileName = argv[1];
  std::string rhsFileName    = argv[2];

  index_type numSystems = atoi(argv[3]);
  std::cout << "Family mtx file name: " << matrixFileName << ", total number of matrices: " << numSystems << std::endl;
  std::cout << "Family rhs file name: " << rhsFileName << ", total number of RHSes: " << numSystems << std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  ReSolve::matrix::Csr* A = nullptr;

  ReSolve::LinAlgWorkspaceCUDA* workspace_CUDA = new ReSolve::LinAlgWorkspaceCUDA;
  workspace_CUDA->initializeHandles();
  ReSolve::MatrixHandler* matrix_handler = new ReSolve::MatrixHandler(workspace_CUDA);
  ReSolve::VectorHandler* vector_handler = new ReSolve::VectorHandler(workspace_CUDA);
  real_type*              rhs            = nullptr;
  real_type*              x              = nullptr;

  vector_type* vec_rhs = nullptr;
  vector_type* vec_x   = nullptr;
  vector_type* vec_r   = nullptr;

  ReSolve::LinSolverDirectKLU*        KLU = new ReSolve::LinSolverDirectKLU;
  ReSolve::LinSolverDirectCuSolverRf* Rf  = new ReSolve::LinSolverDirectCuSolverRf();

  real_type res_nrm = 0.0;
  real_type b_nrm   = 0.0;

  // We need them. They hold a POINTER. Don't delete them here. KLU deletes them.
  ReSolve::matrix::Csr* L = nullptr;
  ReSolve::matrix::Csr* U = nullptr;
  index_type*           P = nullptr;
  index_type*           Q = nullptr;

  int status          = 0;
  int status_refactor = 0;

  for (int i = 0; i < numSystems; ++i)
  {
    if (i < 10)
    {
      fileId = "0" + std::to_string(i);
      rhsId  = "0" + std::to_string(i);
    }
    else
    {
      fileId = std::to_string(i);
      rhsId  = std::to_string(i);
    }
    matrixFileNameFull = "";
    rhsFileNameFull    = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull    = rhsFileName + rhsId + ".mtx";
    std::cout << std::endl
              << std::endl
              << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "Reading: " << matrixFileNameFull << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << std::endl;
    // Read first matrix
    std::ifstream mat_file(matrixFileNameFull);
    if (!mat_file.is_open())
    {
      std::cout << "Failed to open file " << matrixFileNameFull << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhsFileNameFull);
    if (!rhs_file.is_open())
    {
      std::cout << "Failed to open file " << rhsFileNameFull << "\n";
      return -1;
    }
    bool is_expand_symmetric = true;
    if (i == 0)
    {
      A = ReSolve::io::createCsrFromFile(mat_file, is_expand_symmetric);

      rhs     = ReSolve::io::createArrayFromFile(rhs_file);
      x       = new real_type[A->getNumRows()];
      vec_rhs = new vector_type(A->getNumRows());
      vec_x   = new vector_type(A->getNumRows());
      vec_r   = new vector_type(A->getNumRows());
    }
    else
    {
      ReSolve::io::updateMatrixFromFile(mat_file, A);
      ReSolve::io::updateArrayFromFile(rhs_file, &rhs);
    }
    // Copy matrix data to device
    A->syncData(ReSolve::memory::DEVICE);

    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows() << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    // Update host and device data.
    if (i < 2)
    {
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    }
    else
    {
      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
    }
    std::cout << "CSR matrix loaded. Expanded NNZ: " << A->getNnz() << std::endl;

    // Now call direct solver
    if (i < 2)
    {
      KLU->setup(A);
      status = KLU->analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = KLU->factorize();
      std::cout << "KLU factorization status: " << status << std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;
      if (i == 1)
      {
        L = (ReSolve::matrix::Csr*) KLU->getLFactorCsr();
        U = (ReSolve::matrix::Csr*) KLU->getUFactorCsr();
        if (L == nullptr)
        {
          std::cout << "ERROR: L factor is null\n";
          continue; // Skip this iteration
        }
        if (U == nullptr)
        {
          std::cout << "ERROR: U factor is null\n";
          continue; // Skip this iteration
        }
        P = KLU->getPOrdering();
        Q = KLU->getQOrdering();
        Rf->setupCsr(A, L, U, P, Q);
        status_refactor = Rf->refactorize();
        std::cout << "Initial Rf refactorization status: " << status_refactor << std::endl;

        // Don't delete L and U here - they are managed by KLU.
        L = nullptr;
        U = nullptr;
      }
    }
    else
    {
      std::cout << "Using cusolver rf" << std::endl;
      status_refactor = Rf->refactorize();
      std::cout << "cusolver rf refactorization status: " << status_refactor << std::endl;
      status = Rf->solve(vec_rhs, vec_x);
      std::cout << "cusolver rf solve status: " << status << std::endl;
    }

    // Make sure vec_r is properly initialized before using it.
    vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

    matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUS_ONE, ReSolve::memory::DEVICE);
    res_nrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));
    b_nrm   = sqrt(vector_handler->dot(vec_rhs, vec_rhs, ReSolve::memory::DEVICE));
    std::cout << "\t2-Norm of the residual: "
              << std::scientific << std::setprecision(16)
              << res_nrm / b_nrm << "\n";

    if (((res_nrm / b_nrm > 1e-7) && (!std::isnan(res_nrm))) || (status_refactor != 0))
    {
      if ((res_nrm / b_nrm > 1e-7))
      {
        std::cout << "\n \t !!! ALERT !!! Residual norm is too large; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";
      }
      else
      {
        std::cout << "\n \t !!! ALERT !!! cuSolverRf crashed; redoing KLU symbolic and numeric factorization. !!! ALERT !!! \n \n";
      }
      KLU->setup(A);
      status = KLU->analyze();
      std::cout << "KLU analysis status: " << status << std::endl;
      status = KLU->factorize();
      std::cout << "KLU factorization status: " << status << std::endl;
      status = KLU->solve(vec_rhs, vec_x);
      std::cout << "KLU solve status: " << status << std::endl;

      vec_rhs->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);
      vec_r->copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::DEVICE);

      matrix_handler->setValuesChanged(true, ReSolve::memory::DEVICE);

      matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUS_ONE, ReSolve::memory::DEVICE);
      res_nrm = sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::DEVICE));

      std::cout << "\t New residual norm: "
                << std::scientific << std::setprecision(16)
                << res_nrm / b_nrm << "\n";

      L = (ReSolve::matrix::Csr*) KLU->getLFactorCsr();
      U = (ReSolve::matrix::Csr*) KLU->getUFactorCsr();

      if (L != nullptr && U != nullptr)
      {
        P = KLU->getPOrdering();
        Q = KLU->getQOrdering();

        Rf->setupCsr(A, L, U, P, Q);
        status_refactor = Rf->refactorize();
        std::cout << "Rf refactorization after KLU redo status: " << status_refactor << std::endl;
      }

      // Don't delete L and U - they are managed by KLU
      L = nullptr;
      U = nullptr;
    }
  } // for (int i = 0; i < numSystems; ++i)

  if (vec_r != nullptr)
  {
    delete vec_r;
    vec_r = nullptr;
  }
  if (vec_x != nullptr)
  {
    delete vec_x;
    vec_x = nullptr;
  }
  if (vec_rhs != nullptr)
  {
    delete vec_rhs;
    vec_rhs = nullptr;
  }

  if (x != nullptr)
  {
    delete[] x;
    x = nullptr;
  }
  if (rhs != nullptr)
  {
    delete[] rhs;
    rhs = nullptr;
  }

  if (Rf != nullptr)
  {
    delete Rf;
    Rf = nullptr;
  }
  if (KLU != nullptr)
  {
    delete KLU;
    KLU = nullptr;
  }

  if (A != nullptr)
  {
    delete A;
    A = nullptr;
  }

  if (matrix_handler != nullptr)
  {
    delete matrix_handler;
    matrix_handler = nullptr;
  }
  if (vector_handler != nullptr)
  {
    delete vector_handler;
    vector_handler = nullptr;
  }

  if (workspace_CUDA != nullptr)
  {
    delete workspace_CUDA;
    workspace_CUDA = nullptr;
  }

  return 0;
}
