#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/SystemSolver.hpp>

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
  ReSolve::LinAlgWorkspaceCpu* workspace = new ReSolve::LinAlgWorkspaceCpu();
  ReSolve::MatrixHandler* matrix_handler = new ReSolve::MatrixHandler(workspace);
  ReSolve::VectorHandler* vector_handler = new ReSolve::VectorHandler(workspace);
  real_type* rhs = nullptr;
  real_type* x   = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;

  ReSolve::SystemSolver* solver = new ReSolve::SystemSolver();

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
      vec_r = new vector_type(A->getNumRows());
    } else {
      ReSolve::io::readAndUpdateMatrix(mat_file, A_coo);
      ReSolve::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows()
              << " x "           << A->getNumColumns()
              << ", nnz: "       << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? "  << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    //Now convert to CSR.
    if (i < 2) { 
      A->updateFromCoo(A_coo, ReSolve::memory::HOST);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
      vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    } else { 
      A->updateFromCoo(A_coo, ReSolve::memory::HOST);
      vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    }
    std::cout<<"COO to CSR completed. Expanded NNZ: "<< A->getNnzExpanded()<<std::endl;
    //Now call direct solver
    solver->setMatrix(A);
    int status;
    if (i < 2){
      // solver->setup(A);
      status = solver->analyze();
      std::cout<<"solver analysis status: "<<status<<std::endl;
      status = solver->factorize();
      std::cout<<"solver factorization status: "<<status<<std::endl;
      status = solver->solve(vec_rhs, vec_x);
      std::cout<<"solver solve status: "<<status<<std::endl;      
    } else {
      status =  solver->refactorize();
      std::cout<<"solver re-factorization status: "<<status<<std::endl;
      status = solver->solve(vec_rhs, vec_x);
      std::cout<<"solver solve status: "<<status<<std::endl;      
    }
    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler->setValuesChanged(true, "cpu");

    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, "csr", "cpu"); 

    std::cout << "\t 2-Norm of the residual: " 
              << std::scientific << std::setprecision(16) 
              << sqrt(vector_handler->dot(vec_r, vec_r, "cpu")) << "\n";
  }

  //now DELETE
  delete A;
  delete solver;
  delete [] x;
  delete [] rhs;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
