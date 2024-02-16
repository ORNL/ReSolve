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

  LinSolverDirectCpuILU0::~LinSolverDirectCpuILU0()
  {
    if (false) { //(is_analysis_successful) {
      delete L_;
      delete U_;
      delete [] diagU_;
      delete [] idxmap_;
    }
  }

  int LinSolverDirectCpuILU0::setup(matrix::Sparse* A,
                                    matrix::Sparse*,
                                    matrix::Sparse*,
                                    index_type*,
                                    index_type*,
                                    vector_type* )
  {
    int error_sum = 1;

    /// @todo Implement method for setting `zero_diagonal_` value.
    zero_diagonal_ = 0.1;

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

    std::cout << "nnzL: " << nnzL << ", nnzU: " << nnzU << "\n";
    std::cout << "rowsL  rowsU\n";
    for(index_type i = 0; i <= N; ++i) {
      std::cout << rowsL[i] << "  " << rowsU[i] << "\n";
    }

    // Crate and initialize L and U factors
    // L_ = new matrix::Csr(N, N, nnzL);
    // U_ = new matrix::Csr(N, N, nnzU);

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
    std::cout << lcount << " ?= " << nnzL << "\n";
    std::cout << ucount << " ?= " << nnzU << "\n";

    // Set values to L and U matrices
    // L_->updateData(rowsL, colsL, valsL, memory::HOST, memory::HOST);
    // U_->updateData(rowsU, colsU, valsU, memory::HOST, memory::HOST);

    // std::cout <<   "Factor L:\n";
    // L_->print();
    // std::cout << "\nFactor U:\n";
    // U_->print();

    // Allocate temporary vector that maps columns to elements in CSR data.
    idxmap_ = new index_type[N];
    std::cout << "\n\nMapping vector initialized: \n";
    for (index_type u = 0; u < N; ++u)
       idxmap_[u] = -1;

    // Factorize (incompletely)
    std::cout << "\n\nTest factorization:\n";
    for (index_type i = 1; i < N; ++i) {
      for (index_type v = rowsL[i]; v < rowsL[i+1]; ++v) {
        index_type k = colsL[v];
        // std::cout << "(" << i << ", " << k << ")\n";
        for (index_type u = rowsU[k]; u < rowsU[k+1]; ++u) {
           idxmap_[colsU[u]] = u;
        }
        // for (index_type u = 0; u < N; ++u)
        //   std::cout <<  idxmap_[u] << " ";
        // std::cout << "\n";
        valsL[v] /= valsU[rowsU[k]];
        // std::cout << "Diag(" << k << ") = " << valsU[rowsU[k]] << "\n";

        std::cout << "L factor:\n";
        for (index_type w = v+1; w < rowsL[i+1]; ++w) {
          index_type j =  idxmap_[colsL[w]];
          std::cout << "(" << i << ", " << j << ")\n";
          if (j == -1)
            continue;
          valsL[w] -= valsL[v]*valsU[j];
        }

        std::cout << "U factor:\n";
        for (index_type w = rowsU[i]; w < rowsU[i+1]; ++w) {
          index_type j =  idxmap_[colsU[w]];
          std::cout << "(" << i << ", " << j << ")\n";
          if (j == -1)
            continue;
          valsU[w] -= valsL[v]*valsU[j];
        }


        for (index_type u = 0; u < N; ++u)
           idxmap_[u] = -1;
      }
    }

    // Set values to L and U matrices
    // L_->setOwnsData(memory::HOST);
    // U_->setOwnsData(memory::HOST);

    is_analysis_successful = true;
    // L_->updateData(rowsL, colsL, valsL, memory::HOST, memory::HOST);
    // U_->updateData(rowsU, colsU, valsU, memory::HOST, memory::HOST);

    // Use hijacking constructor to create L and U factors
    L_ = new matrix::Csr(N, N, nnzL, false, true, &rowsL, &colsL, &valsL, memory::HOST, memory::HOST);
    U_ = new matrix::Csr(N, N, nnzU, false, true, &rowsU, &colsU, &valsU, memory::HOST, memory::HOST);

    std::cout <<   "Factor L:\n";
    auto* L = dynamic_cast<matrix::Csr*>(L_);
    L->print();
    std::cout << "\nFactor U:\n";
    auto* U = dynamic_cast<matrix::Csr*>(U_);
    U->print();

    return error_sum;
  }

  int LinSolverDirectCpuILU0::reset(matrix::Sparse* A)
  {
    int error_sum = 0;
    
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

  matrix::Sparse* LinSolverDirectCpuILU0::getLFactor()
  {
    return L_;
  }

  matrix::Sparse* LinSolverDirectCpuILU0::getUFactor()
  {
    return U_;
  }

}// namespace resolve
