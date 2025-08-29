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

    void SpGEMMCpu::addProductMatrices(matrix::Csr* A, matrix::Csr* B)
    {
      if (!A_)
      {
        A_ = allocateCholmodType(A);
      }
      if (!B_)
      {
        B_ = allocateCholmodType(B);
      }
      copyDataToCholmodType(A, A_);
      copyDataToCholmodType(B, B_);
    }

    void SpGEMMCpu::addSumMatrix(matrix::Csr* D)
    {
      if (!D_)
      {
        D_ = allocateCholmodType(D);
      }
      copyDataToCholmodType(D, D_);
    }

    void SpGEMMCpu::addResultMatrix(matrix::Csr** E_ptr)
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
      cholmod_sparse* E_chol = cholmod_add(C_chol, D_, &alpha_, &beta_, 1, 0, &Common_);

      if (!(*E_ptr_))
      {
        *E_ptr_ = new matrix::Csr((index_type) E_chol->nrow, (index_type) E_chol->ncol, (index_type) E_chol->nzmax);
      }
      else
      {
        (*E_ptr_)->destroyMatrixData(memory::HOST);
      }

      // Previous data must be de-allocated and new data copied
      // Cholmod does not allow for reuse of arrays
      (*E_ptr_)->copyDataFrom(static_cast<index_type*>(E_chol->p),
                              static_cast<index_type*>(E_chol->i),
                              static_cast<real_type*>(E_chol->x),
                              memory::HOST,
                              memory::HOST);
    }

    /**
     * @brief Allocates a CHOLMOD sparse matrix of the same size as the input CSR matrix
     * @param A[in] - Pointer to CSR matrix
     * @return Pointer to the allocated CHOLMOD sparse matrix
     */
    cholmod_sparse* SpGEMMCpu::allocateCholmodType(matrix::Csr* A)
    {
      return cholmod_allocate_sparse((size_t) A->getNumRows(),
                                     (size_t) A->getNumColumns(),
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
    void SpGEMMCpu::copyDataToCholmodType(matrix::Csr* A, cholmod_sparse* A_chol)
    {
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol->p), A->getRowData(memory::HOST), A->getNumRows() + 1);
      mem_.copyArrayHostToHost(
          static_cast<int*>(A_chol->i), A->getColData(memory::HOST), A->getNnz());
      mem_.copyArrayHostToHost(
          static_cast<double*>(A_chol->x), A->getValues(memory::HOST), A->getNnz());
    }
  } // namespace hykkt
} // namespace ReSolve
