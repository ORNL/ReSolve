#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <resolve/LinSolverDirectLUSOL.hpp>
#include <resolve/matrix/Coo.hpp>
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

void usage()
{
  std::cerr << "usage: lusol_lusol.exe MATRIX_FILE RHS_FILE NUM_SYSTEMS [MATRIX_FILE_ID RHS_FILE_ID...]" << std::endl;
  exit(1);
}

int main(int argc, char* argv[])
{

  if (argc <= 4) {
    usage();
  }

  index_type n_systems = atoi(argv[3]);
  if (n_systems == 0) {
    return 0;
  }

  if (argc % 2 != 0 || (argc - 4) / 2 < n_systems) {
    usage();
  }

  std::string matrix_file_prefix = argv[1];
  std::string rhs_file_prefix = argv[2];

  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> workspace(new ReSolve::LinAlgWorkspaceCpu());
  std::unique_ptr<ReSolve::MatrixHandler> matrix_handler(new ReSolve::MatrixHandler(workspace.get()));
  std::unique_ptr<ReSolve::VectorHandler> vector_handler(new ReSolve::VectorHandler(workspace.get()));
  real_type norm_A, norm_x, norm_r; // used for INF norm

  std::unique_ptr<ReSolve::LinSolverDirectLUSOL> lusol(new ReSolve::LinSolverDirectLUSOL);

  // NOTE: this has to be manually managed because of the way readAndUpdateRhs takes its arguments.
  //       fixing this would require the second parameter to be a reference to a pointer and not a
  //       pointer to a pointer
  real_type* rhs;
  std::unique_ptr<ReSolve::matrix::Coo> A;
  std::unique_ptr<vector_type> vec_rhs, vec_r, vec_x;

  for (int system = 0; system < n_systems; system++) {

    std::string matrix_file_path = matrix_file_prefix + argv[system + 4] + ".mtx";
    std::ifstream matrix_file(matrix_file_path);
    if (!matrix_file.is_open()) {
      std::cout << "Failed to open file " << matrix_file_path << "\n";
      return 1;
    }

    std::cout << "Matrix file: " << matrix_file_path << std::endl;

    std::string rhs_file_path = rhs_file_prefix + argv[system + 5] + ".mtx";
    std::ifstream rhs_file(rhs_file_path);
    if (!rhs_file.is_open()) {
      std::cout << "Failed to open file " << rhs_file_path << "\n";
      return 1;
    }
    std::cout << "RHS file: " << rhs_file_path << std::endl;

    if (system == 0) {
      A = std::unique_ptr<ReSolve::matrix::Coo>(ReSolve::io::readMatrixFromFile(matrix_file));
      rhs = ReSolve::io::readRhsFromFile(rhs_file);
      vec_rhs = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));
      vec_r = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));

      vec_x = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));
      vec_x->allocate(ReSolve::memory::HOST);
    } else {
      ReSolve::io::readAndUpdateMatrix(matrix_file, A.get());
      ReSolve::io::readAndUpdateRhs(rhs_file, &rhs);
    }

    matrix_file.close();
    rhs_file.close();

    std::cout << "Finished reading the matrix and rhs, size: " << A->getNumRows()
              << " x " << A->getNumColumns()
              << ", nnz: " << A->getNnz()
              << ", symmetric? " << A->symmetric()
              << ", Expanded? " << A->expanded() << std::endl;

    vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    vec_rhs->setDataUpdated(ReSolve::memory::HOST);

    ReSolve::matrix::expand(*A.get());
    std::cout << "Matrix expansion completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;

    if (lusol->setup(A.get()) != 0) {
      std::cout << "setup failed on matrix " << system << "/" << n_systems << std::endl;
      return 1;
    }

    if (lusol->analyze() != 0) {
      std::cout << "analysis failed on matrix " << system << "/" << n_systems << std::endl;
      return 1;
    }

    if (lusol->factorize() != 0) {
      std::cout << "factorization failed on matrix " << system << "/" << n_systems << std::endl;
      return 1;
    }

    if (lusol->solve(vec_rhs.get(), vec_x.get()) != 0) {
      std::cout << "solving failed on matrix " << system << "/" << n_systems << std::endl;
      return 1;
    }

    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);

    specializedMatvec(A.get(), vec_x.get(), vec_r.get(), &ONE, &MINUSONE);
    norm_r = vector_handler->infNorm(vec_r.get(), ReSolve::memory::HOST);

    std::cout << "\t2-Norm of the residual: "
              << std::scientific << std::setprecision(16)
              << sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST)) << "\n";
    matrix_handler->matrixInfNorm(A.get(), &norm_A, ReSolve::memory::HOST);
    norm_x = vector_handler->infNorm(vec_x.get(), ReSolve::memory::HOST);
    std::cout << "\tMatrix inf norm: " << std::scientific << std::setprecision(16) << norm_A << "\n"
              << "\tResidual inf norm: " << norm_r << "\n"
              << "\tSolution inf norm: " << norm_x << "\n"
              << "\tNorm of scaled residuals: " << norm_r / (norm_A * norm_x) << "\n";
  }

  delete[] rhs;
  return 0;
}
