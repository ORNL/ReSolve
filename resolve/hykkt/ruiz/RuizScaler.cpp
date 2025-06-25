#include "RuizScaler.hpp"

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
    class RuizScaler;
    class RuizScalingHandler;
  } // namespace hykkt
} // namespace ReSolve

namespace ReSolve
{
  namespace hykkt
  {
    RuizScaler::RuizScaler(index_type          num_iterations,
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
#else
        kernelImpl_ = new RuizScalingKernelsHIP();
#endif
      }

      allocateWorkspace();
    }

    RuizScaler::~RuizScaler()
    {
      deallocateWorkspace();
    }

    void RuizScaler::addHInfo(matrix::Csr* hes)
    {
      hes_i_ = hes->getRowData(memspace_);
      hes_j_ = hes->getColData(memspace_);
      hes_v_ = hes->getValues(memspace_);
    }

    void RuizScaler::addJInfo(matrix::Csr* jac)
    {
      jac_i_ = jac->getRowData(memspace_);
      jac_j_ = jac->getColData(memspace_);
      jac_v_ = jac->getValues(memspace_);
    }

    void RuizScaler::addJtInfo(matrix::Csr* jac_tr)
    {
      jac_tr_i_ = jac_tr->getRowData(memspace_);
      jac_tr_j_ = jac_tr->getColData(memspace_);
      jac_tr_v_ = jac_tr->getValues(memspace_);
    }

    void RuizScaler::addRhsTop(vector::Vector* rhs_top)
    {
      rhs_top_ = rhs_top->getData(memspace_);
    }

    void RuizScaler::addRhsBottom(vector::Vector* rhs_bottom)
    {
      rhs_bottom_ = rhs_bottom->getData(memspace_);
    }

    real_type* RuizScaler::getAggregateScalingVector() const
    {
      return aggregate_scaling_vector_;
    }

    void RuizScaler::scale()
    {
      resetScaling();

      for (index_type i = 0; i < num_iterations_; ++i)
      {
        kernelImpl_->adaptRowMax(n_, total_n_, hes_i_, hes_j_, hes_v_, jac_i_, jac_j_, jac_v_, jac_tr_i_, jac_tr_j_, jac_tr_v_, scaling_vector_);

        kernelImpl_->adaptDiagScale(n_, total_n_, hes_i_, hes_j_, hes_v_, jac_i_, jac_j_, jac_v_, jac_tr_i_, jac_tr_j_, jac_tr_v_, rhs_top_, rhs_bottom_, aggregate_scaling_vector_, scaling_vector_);
      }
    }

    void RuizScaler::resetScaling()
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

    void RuizScaler::allocateWorkspace()
    {
      if (memspace_ == memory::HOST)
      {
        scaling_vector_           = new real_type[n_];
        aggregate_scaling_vector_ = new real_type[total_n_];
      }
      else
      {
        mem_.allocateArrayOnDevice(&scaling_vector_, total_n_);
        mem_.allocateArrayOnDevice(&aggregate_scaling_vector_, total_n_);
      }
      resetScaling();
    }

    void RuizScaler::deallocateWorkspace()
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
