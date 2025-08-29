/**
 * @file SpGEMM.cpp
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 */

#include "SpGEMM.hpp"

#include "SpGEMMCpu.hpp"
#ifdef RESOLVE_USE_CUDA
#include "SpGEMMCuda.hpp"
#elif defined(RESOLVE_USE_HIP)
#include "SpGEMMHip.hpp"
#endif

namespace ReSolve
{
  using real_type = ReSolve::real_type;

  namespace hykkt
  {
    /**
     * Constructor for SpGEMM
     * @param memspace[in] - Memory space for computation.
     * @param alpha[in] - Scalar multiplier for the product.
     * @param beta[in] - Scalar multiplier for the sum.
     */
    SpGEMM::SpGEMM(memory::MemorySpace memspace, real_type alpha, real_type beta)
      : memspace_(memspace)
    {
      if (memspace_ == memory::HOST)
      {
        impl_ = new SpGEMMCpu(alpha, beta);
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        impl_ = new SpGEMMCuda(alpha, beta);
#elif defined(RESOLVE_USE_HIP)
        impl_ = new SpGEMMHip(alpha, beta);
#else
        out::error() << "No GPU support enabled, and memory space set to DEVICE.\n";
        exit(1);
#endif
      }
    }

    /**
     * Destructor for SpGEMM
     */
    SpGEMM::~SpGEMM()
    {
      delete impl_;
    }

    /**
     * Loads the two matrices for the product
     * @param A[in] - Pointer to CSR matrix
     * @param B[in] - Pointer to CSR matrix
     */
    void SpGEMM::addProductMatrices(matrix::Csr* A, matrix::Csr* B)
    {
      impl_->addProductMatrices(A, B);
    }

    /**
     * Loads the sum matrix for the operation
     * @param D[in] - Pointer to CSR matrix
     */
    void SpGEMM::addSumMatrix(matrix::Csr* D)
    {
      impl_->addSumMatrix(D);
    }

    /**
     * Loads the result matrix
     * @param E[in] - Pointer to pointer to CSR matrix
     */
    void SpGEMM::addResultMatrix(matrix::Csr** E_ptr)
    {
      impl_->addResultMatrix(E_ptr);
    }

    /**
     * Computes the result of the SpGEMM operation
     */
    void SpGEMM::compute()
    {
      impl_->compute();
    }
  } // namespace hykkt
} // namespace ReSolve
