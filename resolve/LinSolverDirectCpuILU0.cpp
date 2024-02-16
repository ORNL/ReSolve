/**
 * @file LinSolverDirectCpuILU0.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains definition of a class for incomplete LU factorization on CPU
 * 
 * 
 */
#include <cassert>

#include <resolve/vector/Vector.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <resolve/utilities/logger/Logger.hpp>

#include "LinSolverDirectCpuILU0.hpp"

namespace ReSolve 
{
  LinSolverDirectCpuILU0::LinSolverDirectCpuILU0(LinAlgWorkspaceCpu* workspace)
    : workspace_(workspace)
  {
  }

  /**
   * @brief Destructor
   * 
   * @todo Address how L and U factors are deleted (currently base class does that).
   */
  LinSolverDirectCpuILU0::~LinSolverDirectCpuILU0()
  {
    if (false) { //(owns_factors_) {
      delete L_;
      delete U_;
    }
    delete [] diagU_;
    delete [] idxmap_;
  }

  int LinSolverDirectCpuILU0::setup(matrix::Sparse* A,
                                    matrix::Sparse*,
                                    matrix::Sparse*,
                                    index_type*,
                                    index_type*,
                                    vector_type* )
  {
    int error_sum = 0;

    // Initialize member and local variables.
    A_ = dynamic_cast<matrix::Csr*>(A);
    index_type N = A->getNumRows();
    index_type* rowsA = A->getRowData(memory::HOST);
    index_type* colsA = A->getColData(memory::HOST);
    real_type*  valsA = A->getValues(memory::HOST);

    // Alloacate row pointers
    index_type* rowsU = new index_type[N + 1]{0};
    index_type* rowsL = new index_type[N + 1]{0};

    // Number of nonzeros in factors L and U
    index_type nnzL = 0;
    index_type nnzU = 0;

    // Initialize buffer to store diagonal elements of U
    diagU_ = new real_type[N];
    std::fill(diagU_, diagU_ + N, zero_diagonal_);

    // Find number of nonzeros and row pointers for L and U factors
    bool has_diagonal = false;
    for (index_type i = 0; i < N; ++i) {
      rowsL[i] = nnzL;
      rowsU[i] = nnzU;
      for (index_type j = rowsA[i]; j < rowsA[i+1]; ++j) {
        if (colsA[j] < i) {
          nnzL++;
        } else {
          if (colsA[j] == i) {
            has_diagonal = true;
            diagU_[i] = valsA[j] < zero_diagonal_ ? zero_diagonal_ : valsA[j];
          }
          nnzU++;
        }
      }
      if (has_diagonal) {
        has_diagonal = false;
      } else {
        nnzU++;
        diagU_[i] = zero_diagonal_;
      }
    }
    rowsL[N] = nnzL;
    rowsU[N] = nnzU;

    index_type* colsL = new index_type[nnzL];
    index_type* colsU = new index_type[nnzU];
    real_type* valsL  = new real_type[nnzL];
    real_type* valsU  = new real_type[nnzU];

    // Set data for L and U
    index_type lcount = 0;
    index_type ucount = 0; 
    for (index_type i = 0; i < N; ++i) {
      colsU[ucount] = i;
      valsU[ucount] = diagU_[i];
      ++ucount;
      for (index_type j = rowsA[i]; j < rowsA[i+1]; ++j) {
        if (colsA[j] < i) {
          colsL[lcount] = colsA[j];
          valsL[lcount] = valsA[j];
          ++lcount;
        } 
        if (colsA[j] > i) {
          colsU[ucount] = colsA[j];
          valsU[ucount] = valsA[j];
          ++ucount;
        }
      }
    }
    assert(lcount == nnzL);
    assert(ucount == nnzU);

    // Allocate temporary vector that maps columns to elements in CSR data.
    idxmap_ = new index_type[N];
    for (index_type u = 0; u < N; ++u)
       idxmap_[u] = -1;

    // Factorize (incompletely)
    for (index_type i = 1; i < N; ++i) {
      for (index_type v = rowsL[i]; v < rowsL[i+1]; ++v) {
        index_type k = colsL[v];
        for (index_type u = rowsU[k]; u < rowsU[k+1]; ++u) {
           idxmap_[colsU[u]] = u;
        }
        valsL[v] /= valsU[rowsU[k]];

        for (index_type w = v+1; w < rowsL[i+1]; ++w) {
          index_type j =  idxmap_[colsL[w]];
          if (j == -1)
            continue;
          valsL[w] -= valsL[v]*valsU[j];
        }

        for (index_type w = rowsU[i]; w < rowsU[i+1]; ++w) {
          index_type j =  idxmap_[colsU[w]];
          if (j == -1)
            continue;
          valsU[w] -= valsL[v]*valsU[j];
        }

        for (index_type u = 0; u < N; ++u)
           idxmap_[u] = -1;
      }
    }

    // Use hijacking constructor to create L and U factors
    L_ = new matrix::Csr(N, N, nnzL, false, true, &rowsL, &colsL, &valsL, memory::HOST, memory::HOST);
    U_ = new matrix::Csr(N, N, nnzU, false, true, &rowsU, &colsU, &valsU, memory::HOST, memory::HOST);

    owns_factors_ = true;

    return error_sum;
  }

  int LinSolverDirectCpuILU0::reset(matrix::Sparse* A)
  {
    using namespace memory;
    int error_sum = 0;
    assert(A_->getNumRows() == A->getNumRows());
    assert(A_->getNnz() == A->getNnz());
    A_ = dynamic_cast<matrix::Csr*>(A);

    index_type* rowsL = L_->getRowData(HOST);
    index_type* colsL = L_->getColData(HOST);
    real_type* valsL = L_->getValues(HOST);

    index_type* rowsU = U_->getRowData(HOST);
    index_type* colsU = U_->getColData(HOST);
    real_type* valsU = U_->getValues(HOST);

    // index_type* rowsA = A_->getRowData(HOST);
    index_type* colsA = A_->getColData(HOST);
    real_type* valsA = A_->getValues(HOST);

    // Update values in L and U factors
    index_type N = A_->getNumRows();
    index_type acount = 0; 
    for (index_type i = 0; i < N; ++i) {
      for (index_type j = rowsL[i]; j < rowsL[i+1]; ++j) {
          valsL[j] = valsA[acount];
          ++acount;
      }
      for (index_type j = rowsU[i]; j < rowsU[i+1]; ++j) {
        if ((colsU[j] == i) && (colsA[acount] != i)) {
          valsU[j] = zero_diagonal_;
        } else {
          valsU[j] = valsA[acount];
          ++acount;
        }
      }
    }

    for (index_type u = 0; u < N; ++u)
       idxmap_[u] = -1;

    // Factorize (incompletely)
    for (index_type i = 1; i < N; ++i) {
      for (index_type v = rowsL[i]; v < rowsL[i+1]; ++v) {
        index_type k = colsL[v];
        for (index_type u = rowsU[k]; u < rowsU[k+1]; ++u) {
           idxmap_[colsU[u]] = u;
        }
        valsL[v] /= valsU[rowsU[k]];

        for (index_type w = v+1; w < rowsL[i+1]; ++w) {
          index_type j =  idxmap_[colsL[w]];
          if (j == -1)
            continue;
          valsL[w] -= valsL[v]*valsU[j];
        }

        for (index_type w = rowsU[i]; w < rowsU[i+1]; ++w) {
          index_type j =  idxmap_[colsU[w]];
          if (j == -1)
            continue;
          valsU[w] -= valsL[v]*valsU[j];
        }

        for (index_type u = 0; u < N; ++u)
           idxmap_[u] = -1;
      }
    }

    return error_sum;
  }

  /**
   * @brief Triangular solve
   * 
   * @param[in,out] rhs_vec - right-hand-side vector
   * @return int - error code
   */
  int LinSolverDirectCpuILU0::solve(vector_type* rhs_vec)
  {
    using namespace memory;
    int error_sum = 0;
    assert(A_->getNumRows() == rhs_vec->getSize());

    index_type N = A_->getNumRows();

    real_type* rhs = rhs_vec->getData(HOST);

    index_type* rowsL = L_->getRowData(HOST);
    index_type* colsL = L_->getColData(HOST);
    real_type*  valsL = L_->getValues(HOST);

    // Forward substitution
    for (index_type i = 0; i < N; ++i) {
      for (index_type j = rowsL[i]; j < rowsL[i+1]; ++j) {
        rhs[i] -= valsL[j] * rhs[colsL[j]];
      }
    }

    index_type* rowsU = U_->getRowData(HOST);
    index_type* colsU = U_->getColData(HOST);
    real_type*  valsU = U_->getValues(HOST);

    // Backward substitution
    for (index_type i = N - 1; i >= 0; --i) {
      for (index_type j = rowsU[i] + 1; j < rowsU[i+1]; ++j) {
        rhs[i] -= valsU[j] * rhs[colsU[j]];
      }
      rhs[i] /= valsU[rowsU[i]];
    }

    return error_sum;
  }

  /**
   * @brief Triangular solve
   * 
   * @param[in]  rhs_vec - right-hand-side vector 
   * @param[out] x_vec   - solution vector
   * @return int - status code
   */
  int LinSolverDirectCpuILU0::solve(vector_type* rhs_vec, vector_type* x_vec)
  {
    using namespace memory;
    int error_sum = 0;
    assert(A_->getNumRows() == rhs_vec->getSize());
    assert(A_->getNumRows() == x_vec->getSize());

    const index_type N = A_->getNumRows();

    const real_type* rhs = rhs_vec->getData(HOST);
    real_type*       x   = x_vec->getData(HOST);

    const index_type* rowsL = L_->getRowData(HOST);
    const index_type* colsL = L_->getColData(HOST);
    const real_type*  valsL = L_->getValues(HOST);

    // Forward substitution
    for (index_type i = 0; i < N; ++i) {
      x[i] = rhs[i];
      for (index_type j = rowsL[i]; j < rowsL[i+1]; ++j) {
        x[i] -= valsL[j] * x[colsL[j]];
      }
    }

    const index_type* rowsU = U_->getRowData(HOST);
    const index_type* colsU = U_->getColData(HOST);
    const real_type*  valsU = U_->getValues(HOST);

    // Backward substitution
    for (index_type i = N - 1; i >= 0; --i) {
      for (index_type j = rowsU[i] + 1; j < rowsU[i+1]; ++j) {
        x[i] -= valsU[j] * x[colsU[j]];
      }
      x[i] /= valsU[rowsU[i]];
    }

    return error_sum;
  }

  matrix::Sparse* LinSolverDirectCpuILU0::getLFactor()
  {
    return L_;
  }

  matrix::Sparse* LinSolverDirectCpuILU0::getUFactor()
  {
    return U_;
  }

  /**
   * @brief Sets approximation to zero on matrix diagonal.
   * 
   * If the original matrix has structural zeros on the diagonal, the ILU0
   * analysis will add diagonal elements and set them to `zero_diagonal_`
   * value. The default is 1e-6, this function allows user to change that.
   * 
   * @param z - small value approximating zero
   * @return int - returns status code
   */
  int LinSolverDirectCpuILU0::setZeroDiagonal(real_type z)
  {
    zero_diagonal_ = z;
    return 0;
  }

} // namespace resolve
