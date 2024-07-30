#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
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
using std::chrono::steady_clock;

int specializedInfNorm(ReSolve::matrix::Coo* A,
                       real_type* norm)
{

  std::unique_ptr<index_type[]> sums = std::unique_ptr<index_type[]>(new index_type[A->getNumRows()]);

  index_type* rows = A->getRowData(ReSolve::memory::HOST);
  real_type* values = A->getValues(ReSolve::memory::HOST);

  for (index_type i = 0; i < A->getNnz(); i++) {
    sums[rows[i]] += std::abs(values[i]);
  }

  *norm = *std::max_element(sums.get(), sums.get() + A->getNumRows());
  return 0;
}

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

  bool output_existed = std::filesystem::exists(std::filesystem::path("./lusol_output.csv"));
  std::fstream output("./lusol_output.csv", std::ios::out | std::ios::app);

  if (!output_existed) {
    output << "matrix_file_path,rhs_file_path,system,total_systems,n_rows,n_columns,initial_nnz,cleaned_nnz,ns_factorization,ns_solving,residual_2_norm,matrix_inf_norm,residual_inf_norm,solution_inf_norm,residual_scaled_norm,l_nnz,l_elements,u_nnz,u_elements\n";
  }

  for (int system = 0; system < n_systems; system++) {
    std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> workspace(new ReSolve::LinAlgWorkspaceCpu());
    std::unique_ptr<ReSolve::VectorHandler> vector_handler(new ReSolve::VectorHandler(workspace.get()));
    std::unique_ptr<ReSolve::LinSolverDirectLUSOL> lusol(new ReSolve::LinSolverDirectLUSOL);

    real_type norm_A, norm_x, norm_r; // used for INF norm
    steady_clock clock;

    std::string matrix_file_path = matrix_file_prefix + argv[(2 * system) + 4] + ".mtx";
    std::ifstream matrix_file(matrix_file_path);
    if (!matrix_file.is_open()) {
      std::cout << "Failed to open file " << matrix_file_path << "\n";
      continue;
    }

    std::cout << "Matrix file: " << matrix_file_path << std::endl;

    std::string rhs_file_path = rhs_file_prefix + argv[(2 * system) + 5] + ".mtx";
    std::ifstream rhs_file(rhs_file_path);
    if (!rhs_file.is_open()) {
      std::cout << "Failed to open file " << rhs_file_path << "\n";
      continue;
    }
    std::cout << "RHS file: " << rhs_file_path << std::endl;

    std::unique_ptr<ReSolve::matrix::Coo> A_unexpanded = std::unique_ptr<ReSolve::matrix::Coo>(ReSolve::io::readMatrixFromFile(matrix_file));
    std::unique_ptr<ReSolve::matrix::Coo> A = std::unique_ptr<ReSolve::matrix::Coo>(new ReSolve::matrix::Coo(A_unexpanded->getNumRows(),
                                                                                                             A_unexpanded->getNumColumns(),
                                                                                                             0));
    real_type* rhs = ReSolve::io::readRhsFromFile(rhs_file);
    std::unique_ptr<vector_type> vec_rhs = std::unique_ptr<vector_type>(new vector_type(A_unexpanded->getNumRows()));
    std::unique_ptr<vector_type> vec_r = std::unique_ptr<vector_type>(new vector_type(A_unexpanded->getNumRows()));

    std::unique_ptr<vector_type> vec_x = std::unique_ptr<vector_type>(new vector_type(A_unexpanded->getNumRows()));
    vec_x->allocate(ReSolve::memory::HOST);

    matrix_file.close();
    rhs_file.close();

    std::cout << "Finished reading the matrix and rhs, size: " << A_unexpanded->getNumRows()
              << " x " << A_unexpanded->getNumColumns()
              << ", nnz: " << A_unexpanded->getNnz()
              << ", symmetric? " << A_unexpanded->symmetric()
              << ", Expanded? " << A_unexpanded->expanded() << std::endl;

    vec_rhs->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
    vec_rhs->setDataUpdated(ReSolve::memory::HOST);

    ReSolve::matrix::coo2coo(A_unexpanded.get(), A.get(), ReSolve::memory::HOST);
    std::cout << "Matrix expansion completed. Expanded NNZ: " << A->getNnzExpanded() << std::endl;

    if (lusol->setup(A.get()) != 0) {
      std::cout << "setup failed on matrix " << system + 1 << "/" << n_systems << std::endl;
      continue;
    }

    if (lusol->analyze() != 0) {
      std::cout << "analysis failed on matrix " << system + 1 << "/" << n_systems << std::endl;
      continue;
    }

    steady_clock::time_point factorization_start = clock.now();

    if (lusol->factorize() != 0) {
      std::cout << "factorization failed on matrix " << system + 1 << "/" << n_systems << std::endl;
      continue;
    }

    steady_clock::time_point factorization_end = clock.now();

    steady_clock::duration factorization_time = factorization_end - factorization_start;

    std::cout << "factorized in " << std::chrono::nanoseconds(factorization_time).count() << "ns" << std::endl;

    auto L = lusol->getLFactor();
    auto U = lusol->getUFactor();

    steady_clock::time_point solving_start = clock.now();

    if (lusol->solve(vec_rhs.get(), vec_x.get()) != 0) {
      std::cout << "solving failed on matrix " << system + 1 << "/" << n_systems << std::endl;
      continue;
    }

    steady_clock::time_point solving_end = clock.now();

    steady_clock::duration solving_time = solving_end - solving_start;

    std::cout << "solved in " << std::chrono::nanoseconds(solving_time).count() << "ns" << std::endl;

    vec_r->update(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);

    specializedMatvec(A.get(), vec_x.get(), vec_r.get(), &ONE, &MINUSONE);
    norm_r = vector_handler->infNorm(vec_r.get(), ReSolve::memory::HOST);

    specializedInfNorm(A.get(), &norm_A);
    norm_x = vector_handler->infNorm(vec_x.get(), ReSolve::memory::HOST);

    output << std::setprecision(16) << std::scientific
           << matrix_file_path << ","
           << rhs_file_path << ","
           << system + 1 << ","
           << n_systems << ","
           << A->getNumRows() << ","
           << A->getNumColumns() << ","
           << A_unexpanded->getNnz() << ","
           << A->getNnz() << ","
           << std::chrono::nanoseconds(factorization_time).count() << ","
           << std::chrono::nanoseconds(solving_time).count() << ","
           << sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST)) << ","
           << norm_A << ","
           << norm_r << ","
           << norm_x << ","
           << (norm_r / (norm_A * norm_x)) << ","
           << L->getNnz() << ","
           << (L->getNumRows() * L->getNumColumns()) << ","
           << U->getNnz() << ","
           << (U->getNumRows() * U->getNumColumns()) << std::endl;

    delete[] rhs;
  }

  output.close();
  return 0;
}
