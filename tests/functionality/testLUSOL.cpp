/**
 * @file testLUSOL.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @author Kaleb Brunhoeber
 * @brief Functionality test for LUSOL direct solver
 *
 */
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
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

using namespace ReSolve::constants;
using namespace ReSolve::colors;
using index_type  = ReSolve::index_type;
using real_type   = ReSolve::real_type;
using vector_type = ReSolve::vector::Vector;

// Prototype for Coo to CSR matrix conversion function
static int coo2csr(ReSolve::matrix::Coo* A_coo, ReSolve::matrix::Csr* A_csr, ReSolve::memory::MemorySpace memspace);

int main(int argc, char* argv[])
{
  int status;
  int error_sum = 0;

  // argv[1] contains the path to the data directory, if it is not ./
  const std::string data_path = (argc == 2) ? argv[1] : "./";

  ReSolve::LinAlgWorkspaceCpu   workspace;
  ReSolve::MatrixHandler        matrix_handler(&workspace);
  ReSolve::VectorHandler        vector_handler(&workspace);
  ReSolve::LinSolverDirectLUSOL lusol;

  std::string matrix_one_path = data_path + "/data/matrix_ACTIVSg200_AC_10.mtx";
  std::string matrix_two_path = data_path + "/data/matrix_ACTIVSg200_AC_11.mtx";

  std::string rhs_one_path = data_path + "/data/rhs_ACTIVSg200_AC_10.mtx.ones";
  std::string rhs_two_path = data_path + "/data/rhs_ACTIVSg200_AC_11.mtx.ones";

  std::ifstream matrix_file(matrix_one_path);
  if (!matrix_file.is_open())
  {
    std::cout << "Failed to open " << matrix_one_path << "\n";
    return 1;
  }

  bool                                  is_expand_symmetric = true;
  std::unique_ptr<ReSolve::matrix::Coo> A(ReSolve::io::createCooFromFile(matrix_file, is_expand_symmetric));
  matrix_file.close();

  std::ifstream rhs_file(rhs_one_path);
  if (!rhs_file.is_open())
  {
    std::cout << "Failed to open " << rhs_one_path << "\n";
    return 1;
  }

  real_type* rhs = ReSolve::io::createArrayFromFile(rhs_file);
  rhs_file.close();

  real_type*  x = new real_type[A->getNumRows()];
  vector_type vec_rhs(A->getNumRows());
  vector_type vec_x(A->getNumRows());
  vector_type vec_r(A->getNumRows());

  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs.setDataUpdated(ReSolve::memory::HOST);
  vec_x.allocate(ReSolve::memory::HOST);

  error_sum += lusol.setup(A.get());
  error_sum += lusol.analyze();

  status = lusol.factorize();
  if (status != 0)
  {
    // LUSOL will segfault if solving is attempted after factorization failed
    error_sum += status;
    return error_sum;
  }

  status = lusol.solve(&vec_rhs, &vec_x);
  if (status != 0)
  {
    error_sum += status;
    return error_sum;
  }

  vector_type vec_test(A->getNumRows());
  vector_type vec_diff(A->getNumRows());

  // The solution is supposed to be a vector with all elements 1.
  real_type* x_data = new real_type[A->getNumRows()];
  std::fill_n(x_data, A->getNumRows(), 1.0);

  vec_test.copyDataFrom(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_r.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff.copyDataFrom(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);

  // Matrix-vector product does not support COO format so we need to convert to CSR
  ReSolve::matrix::Csr A_csr(A->getNumRows(), A->getNumColumns(), A->getNnz(), A->symmetric(), A->expanded());
  error_sum += coo2csr(A.get(), &A_csr, ReSolve::memory::HOST);

  // Tell matrix handler this is a new matrix
  matrix_handler.setValuesChanged(true, ReSolve::memory::HOST);

  // Compute r := A*x - r with computed solution x
  error_sum += matrix_handler.matvec(&A_csr,
                                     &vec_x,
                                     &vec_r,
                                     &ONE,
                                     &MINUS_ONE,
                                     ReSolve::memory::HOST);

  // Compute vector norms
  real_type normRmatrix = sqrt(vector_handler.dot(&vec_r, &vec_r, ReSolve::memory::HOST));
  real_type normXtrue   = sqrt(vector_handler.dot(&vec_x, &vec_x, ReSolve::memory::HOST));
  real_type normB       = sqrt(vector_handler.dot(&vec_rhs, &vec_rhs, ReSolve::memory::HOST));

  // Compute vec_diff := vec_diff - vec_x
  vector_handler.axpy(&MINUS_ONE, &vec_x, &vec_diff, ReSolve::memory::HOST);
  // Compute norm of vec_diff
  real_type normDiffMatrix = sqrt(vector_handler.dot(&vec_diff, &vec_diff, ReSolve::memory::HOST));

  // Compute residual r := A*x - r using exact solution x
  vec_r.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  error_sum += matrix_handler.matvec(&A_csr,
                                     &vec_test,
                                     &vec_r,
                                     &ONE,
                                     &MINUS_ONE,
                                     ReSolve::memory::HOST);
  real_type exactSol_normRmatrix = sqrt(vector_handler.dot(&vec_r, &vec_r, ReSolve::memory::HOST));

  std::cout << "Results: \n";
  std::cout << std::scientific << std::setprecision(16);
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix / normB << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix / normXtrue << " (scaled solution error)\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix << " (control; residual norm with exact solution)\n\n";

  delete[] rhs;    // rhs    = nullptr;
  delete[] x;      // x      = nullptr;
  delete[] x_data; // x_data = nullptr;
  real_type scaled_residual_norm_one = normRmatrix / normB;

  //
  // Repeat the test for a different matrix
  //

  matrix_file = std::ifstream(matrix_two_path);
  if (!matrix_file.is_open())
  {
    std::cout << "Failed to open " << matrix_two_path << "\n";
    return 1;
  }

  A = std::unique_ptr<ReSolve::matrix::Coo>(ReSolve::io::createCooFromFile(matrix_file));
  matrix_file.close();

  rhs_file = std::ifstream(rhs_two_path);
  if (!rhs_file.is_open())
  {
    std::cout << "Failed to open " << rhs_two_path << "\n";
    return 1;
  }

  rhs = ReSolve::io::createArrayFromFile(rhs_file);
  rhs_file.close();

  x = new real_type[A->getNumRows()];

  vec_rhs.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_rhs.setDataUpdated(ReSolve::memory::HOST);
  vec_x.allocate(ReSolve::memory::HOST);

  error_sum += lusol.setup(A.get());
  error_sum += lusol.analyze();

  status = lusol.factorize();
  if (status != 0)
  {
    // LUSOL will segfault if solving is attempted after factorization failed
    error_sum += status;
    return error_sum;
  }

  status = lusol.solve(&vec_rhs, &vec_x);
  if (status != 0)
  {
    error_sum += status;
    return error_sum;
  }

  // The solution is supposed to be a vector with all elements 1.
  x_data = new real_type[A->getNumRows()];
  std::fill_n(x_data, A->getNumRows(), 1.0);

  vec_test.copyDataFrom(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_r.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  vec_diff.copyDataFrom(x_data, ReSolve::memory::HOST, ReSolve::memory::HOST);

  // Matrix-vector product does not support COO format so we need to convert to CSR
  error_sum += coo2csr(A.get(), &A_csr, ReSolve::memory::HOST);

  // Tell matrix handler this is a new matrix
  matrix_handler.setValuesChanged(true, ReSolve::memory::HOST);

  // Compute r := A*x - r with computed solution x
  error_sum += matrix_handler.matvec(&A_csr,
                                     &vec_x,
                                     &vec_r,
                                     &ONE,
                                     &MINUS_ONE,
                                     ReSolve::memory::HOST);

  // Compute vector norms
  normRmatrix = sqrt(vector_handler.dot(&vec_r, &vec_r, ReSolve::memory::HOST));
  normXtrue   = sqrt(vector_handler.dot(&vec_x, &vec_x, ReSolve::memory::HOST));
  normB       = sqrt(vector_handler.dot(&vec_rhs, &vec_rhs, ReSolve::memory::HOST));

  // Compute vec_diff := vec_diff - vec_x
  vector_handler.axpy(&MINUS_ONE, &vec_x, &vec_diff, ReSolve::memory::HOST);
  // Compute norm of vec_diff
  normDiffMatrix = sqrt(vector_handler.dot(&vec_diff, &vec_diff, ReSolve::memory::HOST));

  // compute the residual using exact solution
  vec_r.copyDataFrom(rhs, ReSolve::memory::HOST, ReSolve::memory::HOST);
  // Compute residual r := A*x - r using exact solution x
  error_sum += matrix_handler.matvec(&A_csr,
                                     &vec_test,
                                     &vec_r,
                                     &ONE,
                                     &MINUS_ONE,
                                     ReSolve::memory::HOST);
  // Compute residual error norm
  exactSol_normRmatrix = sqrt(vector_handler.dot(&vec_r, &vec_r, ReSolve::memory::HOST));

  std::cout << "Results: \n";
  std::cout << std::scientific << std::setprecision(16);
  std::cout << "\t ||b-A*x||_2                 : " << normRmatrix << " (residual norm)\n";
  std::cout << "\t ||b-A*x||_2/||b||_2         : " << normRmatrix / normB << " (scaled residual norm)\n";
  std::cout << "\t ||x-x_true||_2              : " << normDiffMatrix << " (solution error)\n";
  std::cout << "\t ||x-x_true||_2/||x_true||_2 : " << normDiffMatrix / normXtrue << " (scaled solution error)\n";
  std::cout << "\t ||b-A*x_exact||_2           : " << exactSol_normRmatrix << " (control; residual norm with exact solution)\n\n";

  delete[] rhs;
  delete[] x;
  delete[] x_data;
  real_type scaled_residual_norm_two = normRmatrix / normB;

  if (!std::isfinite(scaled_residual_norm_one) || !std::isfinite(scaled_residual_norm_two))
  {
    std::cout << "Result is not a finite number!\n";
    error_sum++;
  }
  real_type tol = 100 * ReSolve::constants::MACHINE_EPSILON;
  if ((scaled_residual_norm_one > tol) || (scaled_residual_norm_two > tol))
  {
    std::cout << "Result inaccurate!\n";
    error_sum++;
  }
  if (error_sum == 0)
  {
    std::cout << "Test LUSOL " << GREEN << "PASSED" << CLEAR << "\n\n";
  }
  else
  {
    std::cout << "Test LUSOL " << RED << "FAILED" << CLEAR << ", error sum: " << error_sum << "\n\n";
  }

  return error_sum;
}

/**
 * @brief
 *
 * @param A_coo - Input COO matrix without duplicates sorted in row-major order
 * @param A_csr - Output CSR matrix
 * @param memspace - memory space in the output matrix where the data is copied
 * @return int
 *
 * @pre A_coo and A_csr matrices must be of the same size and type.
 */
int coo2csr(ReSolve::matrix::Coo* A_coo, ReSolve::matrix::Csr* A_csr, ReSolve::memory::MemorySpace memspace)
{
  index_type n            = A_coo->getNumRows();
  index_type m            = A_coo->getNumColumns();
  index_type nnz          = A_coo->getNnz();
  bool       is_symmetric = A_coo->symmetric();
  bool       is_expanded  = A_coo->expanded();

  // First make sure the input is correct or the test fails.
  if (n != A_csr->getNumRows() || m != A_csr->getNumColumns() || nnz != A_csr->getNnz() || is_symmetric != A_csr->symmetric() || is_expanded != A_csr->expanded())
  {
    std::cout << "COO and CSR matrices don't match!\n";
    return 1;
  }

  /* const */ index_type* rows_coo = A_coo->getRowData(ReSolve::memory::HOST);
  /* const */ index_type* cols_coo = A_coo->getColData(ReSolve::memory::HOST);
  /* const */ real_type*  vals_coo = A_coo->getValues(ReSolve::memory::HOST);
  index_type*             row_csr  = new index_type[n + 1];
  row_csr[0]                       = 0;
  index_type i_csr                 = 0;
  for (index_type i = 1; i < nnz; ++i)
  {
    if (rows_coo[i] != rows_coo[i - 1])
    {
      i_csr++;
      row_csr[i_csr] = i;
    }
  }
  row_csr[n] = nnz;
  A_csr->copyDataFrom(row_csr, cols_coo, vals_coo, ReSolve::memory::HOST, memspace);

  delete[] row_csr;

  return 0;
}
