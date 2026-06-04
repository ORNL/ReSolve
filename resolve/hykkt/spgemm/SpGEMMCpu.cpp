/**
 * @file SpGEMMCpu.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 * @brief Implementation of SpGEMM using CHOLMOD for CPU
 */

#include "SpGEMMCpu.hpp"

namespace ReSolve
{
  using real_type = ReSolve::real_type;

  namespace hykkt
  {
    /**
     * Constructor for SpGEMMCpu
     * @param alpha[in] - Scalar multiplier for the product.
     * @param beta[in] - Scalar multiplier for the sum.
     */
    SpGEMMCpu::SpGEMMCpu(real_type alpha, real_type beta)
      : alpha_(alpha), beta_(beta)
    {
      cholmod_start(&Common_);

      A_ = nullptr;
      B_ = nullptr;
      D_ = nullptr;
    }

    /**
     * Destructor for SpGEMMCpu
     */
    SpGEMMCpu::~SpGEMMCpu()
    {
      if (A_)
      {
        cholmod_free_sparse(&A_, &Common_);
      }
      if (B_)
      {
        cholmod_free_sparse(&B_, &Common_);
      }
      if (D_)
      {
        cholmod_free_sparse(&D_, &Common_);
      }
      cholmod_finish(&Common_);
    }

    void SpGEMMCpu::loadProductMatrices(matrix::Csr* A, matrix::Csr* B)
    {
      if (!A_)
      {
        A_ = allocateCholmodType(A);
      }
      if (!B_)
      {
        B_ = allocateCholmodType(B);
      }
      copyDataCholmodType(A, A_);
      copyDataCholmodType(B, B_);
    }

    void SpGEMMCpu::loadSumMatrix(matrix::Csr* D)
    {
      if (!D_)
      {
        D_ = allocateCholmodType(D);
      }
      copyDataCholmodType(D, D_);
    }

    void SpGEMMCpu::loadResultMatrix(matrix::Csr** E_ptr)
    {
      E_ptr_ = E_ptr;
    }

    /**
     * @brief Computes the result of the SpGEMM operation
     *
     * Does not store partial results for reuse, as is this is not supported
     * by the CHOLMOD package.
     */
    void SpGEMMCpu::compute()
    {
      cholmod_sparse* C_chol = cholmod_ssmult(B_, A_, 0, 1, 0, &Common_);
      // B_ and A_ are reversed because cholmod_sparse is a CSC matrix
      // and (B^TA^T)^T=AB.
      cholmod_sparse* E_chol = cholmod_add(C_chol, D_, &alpha_, &beta_, 1, 0, &Common_);

      int*       chol_row_data = static_cast<int*>(E_chol->p);
      int*       chol_col_data = static_cast<int*>(E_chol->i);
      real_type* chol_val_data = static_cast<real_type*>(E_chol->x);

      const index_type num_rows = static_cast<index_type>(E_chol->ncol);
      const index_type num_cols = static_cast<index_type>(E_chol->nrow);
      const index_type nnz      = static_cast<index_type>(chol_row_data[E_chol->ncol]);

      if (*E_ptr_)
      {
        delete *E_ptr_;
        *E_ptr_ = nullptr;
      }

      *E_ptr_ = new matrix::Csr(num_rows, num_cols, nnz);
      (*E_ptr_)->allocateMatrixData(memory::HOST);

      index_type* row_data = (*E_ptr_)->getRowData(memory::HOST);
      index_type* col_data = (*E_ptr_)->getColData(memory::HOST);
      real_type*  val_data = (*E_ptr_)->getValues(memory::HOST);

      for (index_type i = 0; i <= num_rows; ++i)
      {
        row_data[i] = static_cast<index_type>(chol_row_data[i]);
      }
      for (index_type i = 0; i < nnz; ++i)
      {
        col_data[i] = static_cast<index_type>(chol_col_data[i]);
        val_data[i] = chol_val_data[i];
      }
      // Previous data must be de-allocated and the new data must be copied.
      // Cholmod does not allow for reuse of arrays.
      (*E_ptr_)->setUpdated(memory::HOST);

      cholmod_free_sparse(&C_chol, &Common_);
      cholmod_free_sparse(&E_chol, &Common_);
    }

    /**
     * @brief Allocates a CHOLMOD sparse matrix of the same size as the input CSR matrix
     * @param A[in] - Pointer to CSR matrix
     * @return Pointer to the allocated CHOLMOD sparse matrix
     */
    cholmod_sparse* SpGEMMCpu::allocateCholmodType(matrix::Csr* A)
    {
      return cholmod_allocate_sparse((size_t) A->getNumColumns(),
                                     (size_t) A->getNumRows(),
                                     (size_t) A->getNnz(),
                                     1,
                                     1,
                                     0,
                                     CHOLMOD_REAL,
                                     &Common_);
    }

    /**
     * Copies the data from a CSR matrix to a CHOLMOD sparse matrix
     * @param A[in] - Pointer to CSR matrix
     * @param A_chol[in] - Pointer to CHOLMOD sparse matrix
     */
    void SpGEMMCpu::copyDataCholmodType(matrix::Csr* A, cholmod_sparse* A_chol)
    {
      int*       chol_row_data = static_cast<int*>(A_chol->p);
      int*       chol_col_data = static_cast<int*>(A_chol->i);
      real_type* chol_val_data = static_cast<real_type*>(A_chol->x);

      index_type* row_data = A->getRowData(memory::HOST);
      index_type* col_data = A->getColData(memory::HOST);
      real_type*  val_data = A->getValues(memory::HOST);

      for (index_type i = 0; i < A->getNumRows() + 1; ++i)
      {
        chol_row_data[i] = static_cast<int>(row_data[i]);
      }

      for (index_type i = 0; i < A->getNnz(); ++i)
      {
        chol_col_data[i] = static_cast<int>(col_data[i]);
        chol_val_data[i] = val_data[i];
      }
    }
  } // namespace hykkt
} // namespace ReSolve
