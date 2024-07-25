#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <resolve/LinSolverDirectKLU.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using std::chrono::steady_clock;

int main(int argc, char* argv[])
{
  // Use the same data types as those you specified in ReSolve build.
  using index_type = ReSolve::index_type;
  using real_type = ReSolve::real_type;
  using vector_type = ReSolve::vector::Vector;

  (void)argc; // TODO: Check if the number of input parameters is correct.
  std::string matrixFileName = argv[1];
  std::string rhsFileName = argv[2];

  index_type numSystems = atoi(argv[3]);
  std::cout << "Family mtx file name: " << matrixFileName << ", total number of matrices: " << numSystems << std::endl;
  std::cout << "Family rhs file name: " << rhsFileName << ", total number of RHSes: " << numSystems << std::endl;

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
  real_type* x = nullptr;

  vector_type* vec_rhs;
  vector_type* vec_x;
  vector_type* vec_r;
  real_type norm_A, norm_x, norm_r; // used for INF norm
  steady_clock clock;
  bool output_existed = std::filesystem::exists(std::filesystem::path("./klu_output.csv"));
  std::fstream output("./klu_output.csv", std::ios::out | std::ios::app);

  if (!output_existed) {
    output << "matrix_file_path,rhs_file_path,system,total_systems,n_rows,n_columns,initial_nnz,cleaned_nnz,ns_factorization,ns_solving,residual_2_norm,matrix_inf_norm,residual_inf_norm,solution_inf_norm,residual_scaled_norm,l_nnz,l_elements,u_nnz,u_elements\n";
  }

  ReSolve::LinSolverDirectKLU* KLU = new ReSolve::LinSolverDirectKLU;

  for (int i = 0; i < numSystems; ++i) {
    index_type j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    // Read matrix first
    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout << std::endl
              << std::endl
              << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "Reading: " << matrixFileNameFull << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << std::endl;
    // Read first matrix
    std::ifstream mat_file(matrixFileNameFull);
    if (!mat_file.is_open()) {
      std::cout << "Failed to open file " << matrixFileNameFull << "\n";
      return -1;
    }
    std::ifstream rhs_file(rhsFileNameFull);
    if (!rhs_file.is_open()) {
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
              << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    // Now convert to CSR.
    A->updateFromCoo(A_coo, ReSolve::memory::HOST);
    vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    std::cout << "COO to CSR completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;
    // Now call direct solver
    int status;

    KLU->setup(A);
    status = KLU->analyze();
    std::cout << "KLU analysis status: " << status << std::endl;

    steady_clock::time_point factorization_start = clock.now();

    status = KLU->factorize();

    steady_clock::time_point factorization_end = clock.now();
    steady_clock::duration factorization_time = factorization_end - factorization_start;

    std::cout << "factorized in " << std::chrono::nanoseconds(factorization_time).count() << "ns" << std::endl;

    std::cout << "KLU factorization status: " << status << std::endl;

    auto L = KLU->getLFactor();
    auto U = KLU->getUFactor();

    steady_clock::time_point solving_start = clock.now();

    status = KLU->solve(vec_rhs, vec_x);

    steady_clock::time_point solving_end = clock.now();
    steady_clock::duration solving_time = solving_end - solving_start;

    std::cout << "solved in " << std::chrono::nanoseconds(solving_time).count() << "ns" << std::endl;

    std::cout << "KLU solve status: " << status << std::endl;

    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

    matrix_handler->matvec(A, vec_x, vec_r, &ONE, &MINUSONE, "csr", ReSolve::memory::HOST);
    norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::HOST);
    matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::HOST);
    norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::HOST);

    output << std::format("{},{},{},{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{},{},{},{}\n",
                          matrixFileNameFull,
                          rhsFileNameFull,
                          i + 1,
                          numSystems,
                          A->getNumRows(),
                          A->getNumColumns(),
                          A_coo->getNnz(),
                          A->getNnz(),
                          std::chrono::nanoseconds(factorization_time).count(),
                          std::chrono::nanoseconds(solving_time).count(),
                          sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST)),
                          norm_A,
                          norm_r,
                          norm_x,
                          norm_r / (norm_A * norm_x),
                          L->getNnz(),
                          L->getNumRows() * L->getNumColumns(),
                          U->getNnz(),
                          U->getNumRows() * U->getNumColumns());
  }

  output.close();

  // now DELETE
  delete A;
  delete KLU;
  delete[] x;
  delete[] rhs;
  delete vec_r;
  delete vec_x;
  delete matrix_handler;
  delete vector_handler;

  return 0;
}
