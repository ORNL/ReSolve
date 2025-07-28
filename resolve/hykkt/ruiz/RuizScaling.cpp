#include "RuizScaling.hpp"

#include "RuizScalingKernelsCpu.hpp"
#ifdef RESOLVE_USE_CUDA
#include "RuizScalingKernelsCuda.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "RuizScalingKernelsHip.hpp"
#endif

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

  using out = ReSolve::io::Logger;

  using namespace ReSolve::constants;

  namespace hykkt
  {
    class RuizScaling;
    class RuizScalingHandler;
  } // namespace hykkt
} // namespace ReSolve

namespace ReSolve
{
  namespace hykkt
  {
    /** Constructor for RuizScaling.
     *  @param num_iterations[in] - Number of iterations for scaling.
     *  @param n[in] - Size of block [1,1]
     *  @param total_n[in] - Total size of the matrix.
     *  @param memspace[in] - Memory space of incoming data and for computation.
     */
    RuizScaling::RuizScaling(index_type          n,
                             index_type          total_n,
                             memory::MemorySpace memspace)
      : n_(n),
        total_n_(total_n),
        memspace_(memspace),
        scaling_vector_(total_n_),
        aggregate_scaling_vector_(total_n_)
    {
      if (memspace_ == memory::HOST)
      {
        kernelImpl_ = new RuizScalingKernelsCpu();
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        kernelImpl_ = new RuizScalingKernelsCuda();
#elif defined(RESOLVE_USE_HIP)
        kernelImpl_ = new RuizScalingKernelsHip();
#else
        out::error() << "RuizScaling: Memory space is DEVICE but no GPU support is enabled.";
        exit(1);
#endif
      }

      scaling_vector_.allocate(memspace_);
      aggregate_scaling_vector_.allocate(memspace_);
      resetScaling();
    }

    RuizScaling::~RuizScaling()
    {
      delete kernelImpl_;
    }

    /**
     *  @brief Add matrix data to the Ruiz scaling handler.
     *  @param hes[in] - Pointer to CSR matrix representing the Hessian.
     *  @param jac[in] - Pointer to CSR matrix representing the Jacobian.
     *  @param jac_tr[in] - Pointer to CSR matrix representing the transposed Jacobian.
     */
    void RuizScaling::addMatrixData(matrix::Csr* hes, matrix::Csr* jac, matrix::Csr* jac_tr)
    {
      hes_    = hes;
      jac_    = jac;
      jac_tr_ = jac_tr;
    }

    /**
     *  @brief Add right-hand side vector to the Ruiz scaling handler.
     *  @param rhs_top[in] - Pointer to the top right-hand side vector.
     *  @param rhs_bottom[in] - Pointer to the bottom right-hand side vector.
     */
    void RuizScaling::addRhsData(vector::Vector* rhs_top, vector::Vector* rhs_bottom)
    {
      rhs_top_    = rhs_top;
      rhs_bottom_ = rhs_bottom;
    }

    /**
     *  @brief Get the scaling vector.
     *  @return Pointer to the scaling vector.
     */
    vector::Vector* RuizScaling::getAggregateScalingVector()
    {
      return &aggregate_scaling_vector_;
    }

    /**
     *  @brief Compute the Ruiz scaling in-place.
     */
    void RuizScaling::scale(index_type num_iterations)
    {
      resetScaling();

      for (index_type i = 0; i < num_iterations; ++i)
      {
        kernelImpl_->adaptRowMax(n_, total_n_, hes_, jac_, jac_tr_, &scaling_vector_);
        kernelImpl_->adaptDiagScale(n_, total_n_, hes_, jac_, jac_tr_, rhs_top_, rhs_bottom_, &aggregate_scaling_vector_, &scaling_vector_);
      }

      // Make the client aware that the data has been updated
      hes_->setUpdated(memspace_);
      jac_->setUpdated(memspace_);
      jac_tr_->setUpdated(memspace_);
      rhs_top_->setDataUpdated(memspace_);
      rhs_bottom_->setDataUpdated(memspace_);
      aggregate_scaling_vector_.setDataUpdated(memspace_);
    }

    /**
     *  @brief Reset the scaling vector to its initial state.
     */
    void RuizScaling::resetScaling()
    {
      aggregate_scaling_vector_.setToConst(ONE, memspace_);
    }
  } // namespace hykkt
} // namespace ReSolve
