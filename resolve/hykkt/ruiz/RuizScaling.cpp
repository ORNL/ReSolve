#include "RuizScaling.hpp"

#include "RuizScalingKernelsCPU.hpp"
#ifdef RESOLVE_USE_CUDA
#include "RuizScalingKernelsCUDA.hpp"
#endif
#ifdef RESOLVE_USE_HIP
#include "RuizScalingKernelsHIP.hpp"
#endif

namespace ReSolve
{
  using index_type = ReSolve::index_type;
  using real_type  = ReSolve::real_type;

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
    RuizScaling::RuizScaling(index_type          num_iterations,
                           index_type          n,
                           index_type          total_n,
                           memory::MemorySpace memspace)
      : num_iterations_(num_iterations), n_(n), total_n_(total_n), memspace_(memspace)
    {
      if (memspace_ == memory::HOST)
      {
        kernelImpl_ = new RuizScalingKernelsCPU();
      }
      else
      {
#ifdef RESOLVE_USE_CUDA
        kernelImpl_ = new RuizScalingKernelsCUDA();
#endif
#ifdef RESOLVE_USE_HIP
        kernelImpl_ = new RuizScalingKernelsHIP();
#endif
      }

      allocateWorkspace();
    }

    RuizScaling::~RuizScaling()
    {
      deallocateWorkspace();
      delete kernelImpl_;
    }

    /**
     *  @brief Add Hessian information to the Ruiz scaling handler.
     *  @param hes[in] - Pointer to CSR matrix representing the Hessian.
     */
    void RuizScaling::addHInfo(matrix::Csr* hes)
    {
      hes_i_ = hes->getRowData(memspace_);
      hes_j_ = hes->getColData(memspace_);
      hes_v_ = hes->getValues(memspace_);
    }

    /**
     *  @brief Add Jacobian information to the Ruiz scaling handler.
     *  @param jac[in] - Pointer to CSR matrix representing the Jacobian.
     */
    void RuizScaling::addJInfo(matrix::Csr* jac)
    {
      jac_i_ = jac->getRowData(memspace_);
      jac_j_ = jac->getColData(memspace_);
      jac_v_ = jac->getValues(memspace_);
    }

    /**
     *  @brief Add Jacobian transpose information to the Ruiz scaling handler.
     *  @param jac_tr[in] - Pointer to CSR matrix representing the transpose of the Jacobian.
     */
    void RuizScaling::addJtInfo(matrix::Csr* jac_tr)
    {
      jac_tr_i_ = jac_tr->getRowData(memspace_);
      jac_tr_j_ = jac_tr->getColData(memspace_);
      jac_tr_v_ = jac_tr->getValues(memspace_);
    }

    /**
     *  @brief Add right-hand side vector to the Ruiz scaling handler.
     *  @param rhs_top[in] - Pointer to the top right-hand side vector.
     */
    void RuizScaling::addRhsTop(vector::Vector* rhs_top)
    {
      rhs_top_ = rhs_top->getData(memspace_);
    }

    /**
     *  @brief Add right-hand side vector to the Ruiz scaling handler.
     *  @param rhs_bottom[in] - Pointer to the bottom right-hand side vector.
     */
    void RuizScaling::addRhsBottom(vector::Vector* rhs_bottom)
    {
      rhs_bottom_ = rhs_bottom->getData(memspace_);
    }

    /**
     *  @brief Get the scaling vector.
     *  @return Pointer to the scaling vector.
     */
    real_type* RuizScaling::getAggregateScalingVector() const
    {
      return aggregate_scaling_vector_;
    }

    /**
     *  @brief Compute the Ruiz scaling in-place.
     */
    void RuizScaling::scale()
    {
      resetScaling();

      for (index_type i = 0; i < num_iterations_; ++i)
      {
        kernelImpl_->adaptRowMax(n_, total_n_, hes_i_, hes_j_, hes_v_, jac_i_, jac_j_, jac_v_, jac_tr_i_, jac_tr_j_, jac_tr_v_, scaling_vector_);

        kernelImpl_->adaptDiagScale(n_, total_n_, hes_i_, hes_j_, hes_v_, jac_i_, jac_j_, jac_v_, jac_tr_i_, jac_tr_j_, jac_tr_v_, rhs_top_, rhs_bottom_, aggregate_scaling_vector_, scaling_vector_);
      }
    }

    /**
     *  @brief Reset the scaling vector to its initial state.
     */
    void RuizScaling::resetScaling()
    {
      if (memspace_ == memory::HOST)
      {
        mem_.setArrayToConstOnHost(aggregate_scaling_vector_, ONE, total_n_);
      }
      else
      {
        mem_.setArrayToConstOnDevice(aggregate_scaling_vector_, ONE, total_n_);
      }
    }

    /**
     *  @brief Allocate memory for the scaling vectors and set initial values.
     */
    void RuizScaling::allocateWorkspace()
    {
      if (memspace_ == memory::HOST)
      {
        scaling_vector_           = new real_type[total_n_];
        aggregate_scaling_vector_ = new real_type[total_n_];
      }
      else
      {
        mem_.allocateArrayOnDevice(&scaling_vector_, total_n_);
        mem_.allocateArrayOnDevice(&aggregate_scaling_vector_, total_n_);
      }
      resetScaling();
    }

    /**
     *  @brief Deallocate memory for the scaling vectors.
     */
    void RuizScaling::deallocateWorkspace()
    {
      if (memspace_ == memory::HOST)
      {
        delete[] scaling_vector_;
        delete[] aggregate_scaling_vector_;
      }
      else
      {
        mem_.deleteOnDevice(scaling_vector_);
        mem_.deleteOnDevice(aggregate_scaling_vector_);
      }
    }
  } // namespace hykkt
} // namespace ReSolve
