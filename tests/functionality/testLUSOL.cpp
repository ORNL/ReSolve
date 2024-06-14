#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <resolve/LinSolverDirectLUSOL.hpp>
#include <resolve/matrix/Coo.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using real_type = ReSolve::real_type;
using vector_type = ReSolve::vector::Vector;

real_type compute_error(int& error_sum,
                        std::unique_ptr<ReSolve::MatrixHandler>& matrix_handler,
                        std::unique_ptr<ReSolve::VectorHandler>& vector_handler,
                        std::string matrix_path,
                        std::string rhs_path)
{
  int status;

  std::ifstream matrix_file(matrix_path);
  if (!matrix_file.is_open()) {
    std::cout << "Failed to open " << matrix_path << "\n";
    return 1;
  }

  std::unique_ptr<ReSolve::matrix::Coo> A(ReSolve::io::readMatrixFromFile(matrix_file));
  matrix_file.close();

  A->expand();

  std::ifstream rhs_file(rhs_path);
  if (!rhs_file.is_open()) {
    std::cout << "Failed to open " << rhs_path << "\n";
    return 1;
  }

  std::unique_ptr<real_type[]> rhs(ReSolve::io::readRhsFromFile(rhs_file));
  rhs_file.close();

  ReSolve::LinSolverDirectLUSOL lusol;
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
    // LUSOL will segfault if you continue to try to solve if factorization failed
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

  std::unique_ptr<real_type[]> x_data(new real_type[A->getNumRows()]);
  std::fill_n(x_data.get(), A->getNumRows(), 0);

  vec_test->setData(x_data.get(), ReSolve::memory::HOST);
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff->update(x_data.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);

  matrix_handler->setValuesChanged(true, ReSolve::memory::HOST);
  error_sum += matrix_handler->matvec(A.get(),
                                      vec_x.get(),
                                      vec_r.get(),
                                      &ONE,
                                      &MINUSONE,
                                      "coo",
                                      ReSolve::memory::HOST);

  real_type normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  real_type normXtrue = sqrt(vector_handler->dot(vec_x.get(), vec_x.get(), ReSolve::memory::HOST));
  real_type normB = sqrt(vector_handler->dot(vec_rhs.get(), vec_rhs.get(), ReSolve::memory::HOST));

  vector_handler->axpy(&MINUSONE, vec_x.get(), vec_diff.get(), ReSolve::memory::HOST);
  real_type normDiffMatrix = sqrt(vector_handler->dot(vec_diff.get(), vec_diff.get(), ReSolve::memory::HOST));

  // compute the residual using exact solution
  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);
  error_sum += matrix_handler->matvec(A.get(),
                                      vec_test.get(),
                                      vec_r.get(),
                                      &ONE,
                                      &MINUSONE,
                                      "coo",
                                      ReSolve::memory::HOST);
  real_type exactSol_normRmatrix = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));
  // evaluate the residual ON THE CPU using COMPUTED solution

  vec_r->update(rhs.get(), ReSolve::memory::HOST, ReSolve::memory::HOST);

  error_sum += matrix_handler->matvec(A.get(), vec_x.get(), vec_r.get(), &ONE, &MINUSONE, "coo", ReSolve::memory::HOST);

  real_type normRmatrixCPU = sqrt(vector_handler->dot(vec_r.get(), vec_r.get(), ReSolve::memory::HOST));

  std::cout << "Results: \n";
  std::cout << "\t ||b-A*x||_2                 : " << std::setprecision(16) << normRmatrix << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2  (CPU)          : " << std::setprecision(16) << normRmatrixCPU << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix / normB << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix / normXtrue << " (scaled solution error)\n";
  std::cout << "\t ||x_true||_2                : " << normXtrue << "\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix << " (control; residual norm with exact solution)\n\n";

  return normRmatrix / normB;
}

int main(int argc, char* argv[])
{
  int error_sum = 0;

  // argv[1] contains the path to the data directory, if it is not ./
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  std::unique_ptr<ReSolve::LinAlgWorkspaceCpu> workspace(new ReSolve::LinAlgWorkspaceCpu());
  std::unique_ptr<ReSolve::MatrixHandler> matrix_handler(new ReSolve::MatrixHandler(workspace.get()));
  std::unique_ptr<ReSolve::VectorHandler> vector_handler(new ReSolve::VectorHandler(workspace.get()));

  std::string matrix_one_path = data_path + "data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrix_two_path = data_path + "data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhs_one_path = data_path + "data/rhs_ACTIVSg200_AC_10.mtx";
  std::string rhs_two_path = data_path + "data/rhs_ACTIVSg200_AC_11.mtx.ones";

  real_type scaled_residual_norm_one = compute_error(error_sum, matrix_handler, vector_handler, matrix_one_path, rhs_one_path);
  real_type scaled_residual_norm_two = compute_error(error_sum, matrix_handler, vector_handler, matrix_two_path, rhs_two_path);

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
