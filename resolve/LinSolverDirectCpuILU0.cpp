/**
 * @file LinSolverDirectCpuILU0.cpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Contains definition of a class for incomplete LU factorization on CPU
 * 
 * 
 */
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

  LinSolverDirectCpuILU0::~LinSolverDirectCpuILU0()
  {
  }

  int LinSolverDirectCpuILU0::setup(matrix::Sparse* A,
                                    matrix::Sparse*,
                                    matrix::Sparse*,
                                    index_type*,
                                    index_type*,
                                    vector_type* )
  {
    int error_sum = 1;

    A_ = dynamic_cast<matrix::Csr*>(A);
    index_type N = A->getNumRows();
    index_type* rowsA = A->getRowData(memory::HOST);
    index_type* colsA = A->getColData(memory::HOST);
    real_type*  valsA = A->getValues(memory::HOST);
    index_type* rowsU = new index_type[N + 1]{0};
    index_type* rowsL = new index_type[N + 1]{0};
    index_type nnzL = 0;
    index_type nnzU = 0;
    real_type* diagU = new real_type[N];
    std::fill(diagU, diagU + N, zero_diagonal_);

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
            diagU[i] = valsA[j] < zero_diagonal_ ? zero_diagonal_ : valsA[j];
          }
          nnzU++;
        }
      }
      if (has_diagonal) {
        has_diagonal = false;
      } else {
        nnzU++;
        diagU[i] = zero_diagonal_;
      }
    }
    rowsL[N] = nnzL;
    rowsU[N] = nnzU;

    std::cout << "nnzL: " << nnzL << ", nnzU: " << nnzU << "\n";
    std::cout << "rowsL  rowsU\n";
    for(index_type i = 0; i <= N; ++i) {
      std::cout << rowsL[i] << "  " << rowsU[i] << "\n";
    }

    // Create factors
    L_ = new matrix::Csr(N, N, nnzL);
    U_ = new matrix::Csr(N, N, nnzU);

    // Crate data arrays for L and U factors
    index_type* colsL = new index_type[nnzL];
    index_type* colsU = new index_type[nnzU];
    real_type* valsL  = new real_type[nnzL];
    real_type* valsU  = new real_type[nnzU];

    // Set data for L and U
    index_type lcount = 0;
    index_type ucount = 0; 
    for (index_type i = 0; i < N; ++i) {
      colsU[ucount] = i;
      valsU[ucount] = diagU[i];
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
    std::cout << lcount << " ?= " << nnzL << "\n";
    std::cout << ucount << " ?= " << nnzU << "\n";

    // Set values to L and U matrices
    L_->updateData(rowsL, colsL, valsL, memory::HOST, memory::HOST);
    U_->updateData(rowsU, colsU, valsU, memory::HOST, memory::HOST);

    std::cout <<   "Factor L:\n";
    L_->print();
    std::cout << "\nFactor U:\n";
    U_->print();


    return error_sum;
  }

  int LinSolverDirectCpuILU0::reset(matrix::Sparse* /* A */)
  {
    int error_sum = 1;
    return error_sum;
  }

  // solution is returned in RHS
  int LinSolverDirectCpuILU0::solve(vector_type* /* rhs */)
  {
    int error_sum = 1;
    return error_sum;
  }

  int LinSolverDirectCpuILU0::solve(vector_type* /* rhs */, vector_type* /* x */)
  {
    int error_sum = 1;
    return error_sum;
  }

}// namespace resolve
