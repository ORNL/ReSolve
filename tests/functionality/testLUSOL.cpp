#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <resolve/Common.hpp>
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

/// @brief Specialized matvec implementation for COO matrices
static int specializedMatvec(ReSolve::matrix::Coo* Ageneric,
                             vector_type* vec_x,
                             vector_type* vec_result,
                             const real_type* alpha,
                             const real_type* beta);

int main(int argc, char* argv[])
{
  int status;
  int error_sum = 0;

  // argv[1] contains the path to the data directory, if it is not ./
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> workspace(new ReSolve::LinAlgWorkspaceCpu());
  std::unique_ptr<ReSolve::MatrixHandler> matrix_handler(new ReSolve::MatrixHandler(workspace.get()));
  std::unique_ptr<ReSolve::VectorHandler> vector_handler(new ReSolve::VectorHandler(workspace.get()));
  ReSolve::LinSolverDirectLUSOL lusol;

  std::string matrix_one_path = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrix_two_path = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhs_one_path = data_path + "data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhs_two_path = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  std::ifstream matrix_file(matrix_one_path);
  if (!matrix_file.is_open()) {
    std::cout << "Failed to open " << matrix_one_path << "\n";
    return 1;
  }

  std::unique_ptr<ReSolve::matrix::Coo> A(ReSolve::io::readMatrixFromFile(matrix_file));
  matrix_file.close();

  ReSolve::matrix::expand(*A);

  std::ifstream rhs_file(rhs_one_path);
  if (!rhs_file.is_open()) {
    std::cout << "Failed to open " << rhs_one_path << "\n";
    return 1;
  }

  std::unique_ptr<real_type[]> rhs(ReSolve::io::readRhsFromFile(rhs_file));
  rhs_file.close();

  std::unique_ptr<real_type[]> x(new real_type[A->getNumRows()]);
  std::unique_ptr<vector_type> vec_rhs(new vector_type(A->getNumRows()));
  std::unique_ptr<vector_type> vec_x(new vector_type(A->getNumRows()));
  std::unique_ptr<vector_type> vec_r(new vector_type(A->getNumRows()));

  vec_rhs->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  vec_x->allocate(ReSolve::memory::HOST);

  error_sum += lusol.setup(A.get());
  error_sum += lusol.analyze();

  status = lusol.factorize();
  if (status != 0) {
    // LUSOL will segfault if solving is attempted after factorization failed
    error_sum += status;
    return 1;
  }

  status = lusol.solve(vec_rhs.get(), vec_x.get());
  if (status != 0) {
    error_sum += status;
    return 1;
  }

  std::unique_ptr<vector_type> vec_test(new vector_type(A->getNumRows()));
  std::unique_ptr<vector_type> vec_diff(new vector_type(A->getNumRows()));

  real_type* x_data = new real_type[A->getNumRows()];
  std::fill_n(x_data, A->getNumRows(), 1.0);

  vec_test->setData(x_data, ReSolve::memory::HOST);
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);

  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);
  error_sum += specializedMatvec(A.get(),
                                 vec_x.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);

  real_type normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  real_type normXtrue = sqrt(vector_handler->dot(vec_x.get(), vec_x.get(), ReSolve::memory::HOST));
  real_type normB = sqrt(vector_handler->dot(vec_rhs.get(), vec_rhs.get(), ReSolve::memory::HOST));

  vector_handler->axpy(&MINUSONE, vec_x.get(), vec_diff.get(), ReSolve::memory::HOST);
  real_type normDiffMatrix = sqrt(vector_handler->dot(vec_diff.get(), vec_diff.get(), ReSolve::memory::HOST));

  // compute the residual using exact solution
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  error_sum += specializedMatvec(A.get(),
                                 vec_test.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);
  real_type exactSol_normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  // evaluate the residual ON THE CPU using COMPUTED solution

  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);

  error_sum += specializedMatvec(A.get(),
                                 vec_x.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);

  real_type normRmatrixCPU = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));

  std::cout << "Results: \n";
  std::cout << "\t ||b-A*x||_2                 : " << std::setprecision(16) << normRmatrix << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2  (CPU)          : " << std::setprecision(16) << normRmatrixCPU << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix / normB << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix / normXtrue << " (scaled solution error)\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix << " (control; residual norm with exact solution)\n\n";

  delete[] x_data;
  real_type scaled_residual_norm_one = normRmatrix / normB;

  matrix_file = std::ifstream(matrix_two_path);
  if (!matrix_file.is_open()) {
    std::cout << "Failed to open " << matrix_two_path << "\n";
    return 1;
  }

  A = std::unique_ptr<ReSolve::matrix::Coo>(ReSolve::io::readMatrixFromFile(matrix_file));
  matrix_file.close();

  ReSolve::matrix::expand(*A);

  rhs_file = std::ifstream(rhs_two_path);
  if (!rhs_file.is_open()) {
    std::cout << "Failed to open " << rhs_two_path << "\n";
    return 1;
  }

  rhs = std::unique_ptr<real_type[]>(ReSolve::io::readRhsFromFile(rhs_file));
  rhs_file.close();

  x = std::unique_ptr<real_type[]>(new real_type[A->getNumRows()]);
  vec_rhs = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));
  vec_x = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));
  vec_r = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));

  vec_rhs->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs->setDataUpdated(ReSolve::memory::HOST);
  vec_x->allocate(ReSolve::memory::HOST);

  error_sum += lusol.setup(A.get());
  error_sum += lusol.analyze();

  status = lusol.factorize();
  if (status != 0) {
    // LUSOL will segfault if solving is attempted after factorization failed
    error_sum += status;
    return 1;
  }

  status = lusol.solve(vec_rhs.get(), vec_x.get());
  if (status != 0) {
    error_sum += status;
    return 1;
  }

  vec_test = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));
  vec_diff = std::unique_ptr<vector_type>(new vector_type(A->getNumRows()));

  x_data = new real_type[A->getNumRows()];
  std::fill_n(x_data, A->getNumRows(), 1.0);

  vec_test->setData(x_data, ReSolve::memory::HOST);
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff->update(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);

  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);
  error_sum += specializedMatvec(A.get(),
                                 vec_x.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);

  normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  normXtrue = sqrt(vector_handler->dot(vec_x.get(), vec_x.get(), ReSolve::memory::HOST));
  normB = sqrt(vector_handler->dot(vec_rhs.get(), vec_rhs.get(), ReSolve::memory::HOST));

  vector_handler->axpy(&MINUSONE, vec_x.get(), vec_diff.get(), ReSolve::memory::HOST);
  normDiffMatrix = sqrt(vector_handler->dot(vec_diff.get(), vec_diff.get(), ReSolve::memory::HOST));

  // compute the residual using exact solution
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  error_sum += specializedMatvec(A.get(),
                                 vec_test.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);
  exactSol_normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  // evaluate the residual ON THE CPU using COMPUTED solution

  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);

  error_sum += specializedMatvec(A.get(),
                                 vec_x.get(),
                                 vec_r.get(),
                                 &ONE,
                                 &MINUSONE);

  normRmatrixCPU = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));

  std::cout << "Results: \n";
  std::cout << "\t ||b-A*x||_2                 : " << std::setprecision(16) << normRmatrix << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2  (CPU)          : " << std::setprecision(16) << normRmatrixCPU << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix / normB << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix / normXtrue << " (scaled solution error)\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix << " (control; residual norm with exact solution)\n\n";

  delete[] x_data;
  real_type scaled_residual_norm_two = normRmatrix / normB;

  if (scaled_residual_norm_one > 1e-16 || scaled_residual_norm_two > 1e-16) {
    std::cout << "the result is inaccurate\n";
    error_sum++;
  }

  if (error_sum == 0) {
    std::cout << "PASSED" << std::endl;
  } else {
    std::cout << "FAILED, #errors = " << error_sum << std::endl;
  }

  return error_sum;
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

  // same algorithm used above, adapted to be workable with a coo matrix
  // TODO: optimize this a little more---i'm not too sure how good it is

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
