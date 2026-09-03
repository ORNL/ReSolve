#include "PreconditionerIChol0.hpp"

#include <cassert>

namespace ReSolve
{
  using out = io::Logger;

  /**
   * @brief Constructor for PreconditionerIChol0.
   *
   * @param[in] solver - Pointer to the CholeskyType object.
   */
  PreconditionerIChol0::PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceCpu*)
    : matrix_handler_(matrix_handler)
  {
    out::error() << "Not implemented!";
  }

#ifdef RESOLVE_USE_CUDA
  /**
   * @brief Constructor for PreconditionerIChol0.
   *
   * @param[in] solver - Pointer to the CholeskyType object.
   */
  PreconditionerIChol0::PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceCUDA* workspace)
    : matrix_handler_(matrix_handler)
  {
    impl_ = new PreconditionerIChol0Cuda(workspace);
  }

#elif defined(RESOLVE_USE_HIP)
  /**
   * @brief Constructor for PreconditionerIChol0.
   *
   * @param[in] solver - Pointer to the CholeskyType object.
   */
  PreconditionerIChol0::PreconditionerIChol0(MatrixHandler* matrix_handler, LinAlgWorkspaceHIP* workspace)
    : matrix_handler_(matrix_handler)
  {
    impl_ = new PreconditionerIChol0Hip(workspace);
  }
#endif

  /**
   * @brief Destructor for PreconditionerIChol0
   */
  PreconditionerIChol0::~PreconditionerIChol0()
  {
    delete L_;
  }

  /**
   * @brief Sets up the incomplete Cholesky solver with the given matrix
   *
   * @param[in] A - System matrix to set up the preconditioner with
   *
   * @return int 0 if successful, 1 if it fails
   */
  int PreconditionerIChol0::setup(matrix_type* A)
  {
    if (A == nullptr)
    {
      return 1;
    }
    
    assert(A->getSparseFormat() == matrix_type::COMPRESSED_SPARSE_ROW);

    L_ = new matrix::Csr(A->getNumRows(), A->getNumColumns(), A->getNnz());
    L_->allocateAll(memory::DEVICE); // ANDREW TODO: get memspace
    L_->copyFromExternal(A->getRowData(memory::DEVICE),
                         A->getColData(memory::DEVICE),
                         A->getValues(memory::DEVICE),
                         memory::DEVICE,
                         memory::DEVICE);
    if (numeric_boost_ != 0.0)
    {
      matrix_handler_->addDiag(L_, numeric_boost_, memory::DEVICE);
    }

    return impl_->setup(L_);
  }

  /**
   * @brief Applies the preconditioner to solve the system Mx = rhs
   *
   * Computes x = M^(-1) * rhs where M is the preconditioner matrix.
   *
   * @param[in] rhs - Right-hand-side vector
   * @param[in] x   - Solution vector
   *
   * @return int 0 if successful, 1 if fails
   */
  int PreconditionerIChol0::apply(vector_type* rhs, vector_type* x)
  {
    return impl_->apply(rhs, x);
  }

  /**
   * @brief Resets the preconditioner with the given matrix
   *
   * @param[in] A - System matrix to reset the preconditioner with
   *
   * @return int 0 if successful, 1 if it fails
   */
  int PreconditionerIChol0::reset(matrix_type* A)
  {
    // if (A == nullptr)
    // {
    //   return 1;
    // }

    return 0;
  }

  void PreconditionerIChol0::setNumRhs(index_type num_rhs)
  {
    impl_->setNumRhs(num_rhs);
  }

  // ...
  void PreconditionerIChol0::setNumericBoost(real_type numeric_boost)
  {
    matrix_handler_->addDiag(L_, numeric_boost - numeric_boost_, memory::DEVICE);
    numeric_boost_ = numeric_boost;
  }
}