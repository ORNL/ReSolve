#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <resolve/LinSolverDirectLUSOL.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/Csc.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Utilities.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using index_type = ReSolve::index_type;
using real_type = ReSolve::real_type;
using vector_type = ReSolve::vector::Vector;

int specializedMatvec(ReSolve::matrix::Coo* Ageneric,
                      vector_type* vec_x,
                      vector_type* vec_result,
                      const real_type* alpha,
                      const real_type* beta)
{

  ReSolve::matrix::Coo* A = static_cast<ReSolve::matrix::Coo*>(Ageneric);
  index_type* rows = A->getRowData(ReSolve::memory::HOST);
  index_type* columns = A->getColData(ReSolve::memory::HOST);
  real_type* values = A->getValues(ReSolve::memory::HOST);

  real_type* xs = vec_x->getData(ReSolve::memory::HOST);
  real_type* rs = vec_result->getData(ReSolve::memory::HOST);

  index_type n_rows = A->getNumRows();

  std::unique_ptr<real_type[]> sums(new real_type[n_rows]);
  std::fill_n(sums.get(), n_rows, 0);

  std::unique_ptr<real_type[]> compensations(new real_type[n_rows]);
  std::fill_n(compensations.get(), n_rows, 0);

  real_type y, t;

  for (index_type i = 0; i < A->getNnz(); i++) {
    y = (values[i] * xs[columns[i]]) - compensations[rows[i]];
    t = sums[rows[i]] + y;
    compensations[rows[i]] = t - sums[rows[i]] - y;
    sums[rows[i]] = t;
  }

  for (index_type i = 0; i < n_rows; i++) {
    sums[i] *= *alpha;
    rs[i] = (rs[i] * *beta) + sums[i];
  }

  vec_result->setDataUpdated(ReSolve::memory::HOST);
  return 0;
}

int main(int argc, char* argv[])
{

  if (argc <= 4) {
    std::cerr << "usage: lusol_lusol.exe MATRIX_FILE RHS_FILE NUM_SYSTEMS [MATRIX_FILE_ID RHS_FILE_ID...]" << std::endl;
    return 1;
  }

  std::string matrixFileName = argv[1];
  std::string rhsFileName = argv[2];

  index_type numSystems = atoi(argv[3]);
  if (numSystems == 0) {
    return 0;
  }

  std::cout << "Family mtx file name: " << matrixFileName << ", total number of matrices: " << numSystems << std::endl;
  std::cout << "Family rhs file name: " << rhsFileName << ", total number of RHSes: " << numSystems << std::endl;

  std::string fileId;
  std::string rhsId;
  std::string matrixFileNameFull;
  std::string rhsFileNameFull;

  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> workspace(new ReSolve::LinAlgWorkspaceCpu());
  std::unique_ptr<ReSolve::MatrixHandler> matrix_handler(new ReSolve::MatrixHandler(workspace.get()));
  std::unique_ptr<ReSolve::VectorHandler> vector_handler(new ReSolve::VectorHandler(workspace.get()));
  real_type norm_A, norm_x, norm_r; // used for INF norm

  std::unique_ptr<ReSolve::LinSolverDirectLUSOL> lusol(new ReSolve::LinSolverDirectLUSOL);

  ReSolve::matrix::Coo* A;
  real_type* rhs;
  vector_type *vec_rhs, *vec_r, *vec_x;

  for (int i = 0; i < numSystems; ++i) {
    index_type j = 4 + i * 2;
    fileId = argv[j];
    rhsId = argv[j + 1];

    matrixFileNameFull = "";
    rhsFileNameFull = "";

    matrixFileNameFull = matrixFileName + fileId + ".mtx";
    rhsFileNameFull = rhsFileName + rhsId + ".mtx";
    std::cout << "Reading: " << matrixFileNameFull << std::endl;

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
      A = ReSolve::io::readMatrixFromFile(mat_file);
      rhs = ReSolve::io::readRhsFromFile(rhs_file);
      vec_rhs = new vector_type(A->getNumRows());
      vec_r = new vector_type(A->getNumRows());

      vec_x = new vector_type(A->getNumRows());
      vec_x->allocate(ReSolve::memory::HOST);
    } else {
      ReSolve::io::readAndUpdateMatrix(mat_file, A);
      ReSolve::io::readAndUpdateRhs(rhs_file, &rhs);
    }
    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows()
              << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;
    mat_file.close();
    rhs_file.close();

    vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    vec_rhs->setDataUpdated(ReSolve::memory::HOST);
    ReSolve::matrix::expand(*A);
    std::cout << "Matrix expansion completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;
    // Now call direct solver
    int status;

    lusol->setup(A);
    status = lusol->analyze();
    std::cout << "LUSOL analysis status: " << status << std::endl;
    status = lusol->factorize();
    std::cout << "LUSOL factorization status: " << status << std::endl;
    status = lusol->solve(vec_rhs, vec_x);
    std::cout << "LUSOL solve status: " << status << std::endl;
    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

    specializedMatvec(A, vec_x, vec_r, &ONE, &MINUSONE);
    norm_r = vector_handler->infNorm(vec_r, ReSolve::memory::HOST);

    std::cout << "\t2-Norm of the residual: "
              << std::scientific << std::setprecision(16)
              << sqrt(vector_handler->dot(vec_r, vec_r, ReSolve::memory::HOST)) << "\n";
    matrix_handler->matrixInfNorm(A, &norm_A, ReSolve::memory::HOST);
    norm_x = vector_handler->infNorm(vec_x, ReSolve::memory::HOST);
    std::cout << "\tMatrix inf  norm: " << std::scientific << std::setprecision(16) << norm_A << "\n"
              << "\tResidual inf norm: " << norm_r << "\n"
              << "\tSolution inf norm: " << norm_x << "\n"
              << "\tNorm of scaled residuals: " << norm_r / (norm_A * norm_x) << "\n";
  }

  delete A;
  delete[] rhs;
  delete vec_r;
  delete vec_x;
  delete vec_rhs;

  return 0;
}
