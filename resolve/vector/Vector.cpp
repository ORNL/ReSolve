#include <cassert>
#include <cstring>

#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>

namespace ReSolve
{
  namespace vector
  {

    using out = ReSolve::io::Logger;

    /**
     * @brief Single vector constructor.
     *
     * @param[in] n - Number of elements in the vector
     */
    Vector::Vector(index_type n)
      : n_capacity_(n),
        k_(1),
        n_size_(n),
        gpu_updated_(new bool[1]),
        cpu_updated_(new bool[1])
    {
      gpu_updated_[0] = false;
      cpu_updated_[0] = false;
    }

    /**
     * @brief Multivector constructor.
     *
     * @param[in] n - Number of elements in the vector
     * @param[in] k - Number of vectors in multivector
     */
    Vector::Vector(index_type n, index_type k)
      : n_capacity_(n),
        k_(k),
        n_size_(n),
        gpu_updated_(new bool[k]),
        cpu_updated_(new bool[k])
    {
      setHostUpdated(false);
      setDeviceUpdated(false);
    }

    /**
     * @brief destructor.
     *
     */
    Vector::~Vector()
    {
      if (owns_cpu_data_ && h_data_)
        mem_.deleteOnHost(h_data_);
      if (owns_gpu_data_ && d_data_)
        mem_.deleteOnDevice(d_data_);
      delete[] gpu_updated_;
      delete[] cpu_updated_;
    }

    /**
     * @brief Get capacity of a single vector.
     *
     * Vector memory is allocated to `n_capacity_*k_`. This is the maximum
     * number of elements that the (multi)vector can hold.
     *
     * @return `n_capacity_` the maximum number of elements in the vector.
     */
    index_type Vector::getCapacity() const
    {
      return n_capacity_;
    }

    /**
     * @brief get the number of elements in a single vector.
     *
     * For vectors with changing sizes, set the vector capacity to
     * the maximum expected size.
     *
     * @return `n_size_` number of elements currently in the vector.
     */
    index_type Vector::getSize() const
    {
      return n_size_;
    }

    /**
     * @brief Get the number of vectors in multivector.
     *
     * @return _k_, number of vectors in the multivector,
     * or 1 if the vector is not a multivector.
     */
    index_type Vector::getNumVectors() const
    {
      return k_;
    }

    /**
     * @brief Set the vector data pointer (HOST or DEVICE) to an external data.
     *
     * @param[in] data     - Pointer to data
     * @param[in] memspace - Memory space (HOST or DEVICE)
     *
     * @warning This function DOES NOT ALLOCATE any data, it only assigns the
     * pointer.
     *
     * @warning This is an expert level method. Use only if you know what
     * you are doing.
     */
    int Vector::setData(real_type* data, memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        if (owns_cpu_data_ && h_data_)
        {
          out::error() << "Trying to set vector host values, but the values already set!\n";
          out::error() << "Ignoring setData function call ...\n";
          return 1;
        }
        h_data_ = data;
        setHostUpdated(true);
        setDeviceUpdated(false);
        owns_cpu_data_ = false;
        break;
      case DEVICE:
        if (owns_gpu_data_ && d_data_)
        {
          out::error() << "Trying to set vector device values, but the values already set!\n";
          out::error() << "Ignoring setData function call ...\n";
          return 1;
        }
        d_data_ = data;
        setHostUpdated(false);
        setDeviceUpdated(true);
        owns_gpu_data_ = false;
        break;
      }
      return 0;
    }

    /**
     * @brief Set the flag to indicate that the data (HOST or DEVICE) has been
     * updated.
     *
     * Use this function if you update vector elements by accessing the raw data
     * pointer.
     *
     * @param[in] memspace - Memory space (HOST or DEVICE)
     *
     * @warning This is an expert level method. Use only if you know what
     * you are doing.
     */
    int Vector::setDataUpdated(memory::MemorySpace memspace)
    {
      assert(cpu_updated_ && gpu_updated_ && "Update flags not allocated");

      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        setHostUpdated(true);
        setDeviceUpdated(false);
        break;
      case DEVICE:
        setHostUpdated(false);
        setDeviceUpdated(true);
        break;
      }
      return 0;
    }

    /**
     * @brief Set the flag to indicate that the data (HOST or DEVICE) for
     * vector `j` in the multivector has been updated.
     *
     * Use this function if you update vector elements by accessing the raw data
     * pointer.
     *
     * @param[in] memspace - Memory space (HOST or DEVICE)
     *
     * @warning This is an expert level method. Use only if you know what
     * you are doing.
     */
    int Vector::setDataUpdated(index_type j, memory::MemorySpace memspace)
    {
      assert(cpu_updated_ && gpu_updated_ && "Update flags not allocated");

      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        cpu_updated_[j] = true;
        gpu_updated_[j] = false;
        break;
      case DEVICE:
        gpu_updated_[j] = true;
        cpu_updated_[j] = false;
        break;
      }
      return 0;
    }

    /**
     * @brief Copy data from another vector.
     *
     * @param[in] v           - Vector, which data will be copied
     * @param[in] memspaceIn  - Memory space of the incoming data (HOST or DEVICE)
     * @param[in] memspaceOut - Memory space the data will be copied to (HOST or DEVICE)
     *
     * @pre   size of _v_ is equal or larger than the current vector size.
     */
    int Vector::copyDataFrom(Vector* v, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
    {
      real_type* data = v->getData(memspaceIn);
      return copyDataFrom(data, memspaceIn, memspaceOut);
    }

    /**
     * @brief Copy vector data from input array.
     *
     * This function allocates (if necessary) and copies the data.
     *
     * @param[in] data        - Data that is to be copied
     * @param[in] memspaceIn  - Memory space of the incoming data (HOST or DEVICE)
     * @param[in] memspaceOut - Memory space the data will be copied to (HOST or DEVICE)
     *
     * @return 0 if successful, -1 otherwise.
     */
    int Vector::copyDataFrom(const real_type* data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
    {
      int control = -1;
      if ((memspaceIn == memory::HOST) && (memspaceOut == memory::HOST))
      {
        control = 0;
      }
      if ((memspaceIn == memory::HOST) && (memspaceOut == memory::DEVICE))
      {
        control = 1;
      }
      if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))
      {
        control = 2;
      }
      if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::DEVICE))
      {
        control = 3;
      }

      if ((memspaceOut == memory::HOST) && (h_data_ == nullptr))
      {
        // allocate first
        h_data_        = new real_type[n_capacity_ * k_];
        owns_cpu_data_ = true;
      }
      if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr))
      {
        // allocate first
        mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
        owns_gpu_data_ = true;
      }

      switch (control)
      {
      case 0: // cpu->cpu
        mem_.copyArrayHostToHost(h_data_, data, n_size_ * k_);
        setHostUpdated(true);
        setDeviceUpdated(false);
        break;
      case 2: // gpu->cpu
        mem_.copyArrayDeviceToHost(h_data_, data, n_size_ * k_);
        setHostUpdated(true);
        setDeviceUpdated(false);
        break;
      case 1: // cpu->gpu
        mem_.copyArrayHostToDevice(d_data_, data, n_size_ * k_);
        setHostUpdated(false);
        setDeviceUpdated(true);
        break;
      case 3: // gpu->gpu
        mem_.copyArrayDeviceToDevice(d_data_, data, n_size_ * k_);
        setHostUpdated(false);
        setDeviceUpdated(true);
        break;
      default:
        return -1;
      }
      return 0;
    }

    /**
     * @brief get a pointer to HOST or DEVICE vector data.
     *
     * @param[in] memspace  - Memory space of the pointer (HOST or DEVICE)
     *
     * @return pointer to the vector data (HOST or DEVICE). In case of multivectors,
     * vectors are stored column-wise.
     *
     * @note This function gives you access to the pointer, not to a copy.
     * If you change the values using the pointer, the vector values will change too.
     */
    real_type* Vector::getData(memory::MemorySpace memspace)
    {
      using memory::DEVICE;
      using memory::HOST;

      switch (memspace)
      {
      case HOST:
        if ((cpu_updated_[0] == false) && (gpu_updated_[0] == true))
        {
          syncData(memspace);
        }
        return h_data_;
      case DEVICE:
        if ((gpu_updated_[0] == false) && (cpu_updated_[0] == true))
        {
          syncData(memspace);
        }
        return d_data_;
      default:
        return nullptr;
      }
    }

    /**
     * @brief get a pointer to HOST or DEVICE data of a particular vector in a multivector.
     *
     * @param[in] j         - Index of a vector in multivector
     * @param[in] memspace  - Memory space of the pointer (HOST or DEVICE)
     *
     * @return pointer to the _i_th vector data (HOST or DEVICE) within a multivector.
     *
     * @pre `j` < `k_` i.e, `j` is smaller than the total number of vectors in multivector.
     *
     * @note This function gives you access to the pointer, not to a copy.
     * If you change the values using the pointer, the vector values will change too.
     */
    real_type* Vector::getData(index_type j, memory::MemorySpace memspace)
    {
      using memory::DEVICE;
      using memory::HOST;

      if (k_ <= j)
      {
        out::error() << "Trying to get data for vector " << j << " in multivector"
                     << " but there are only " << k_ << " vectors!\n";
        return nullptr;
      }

      switch (memspace)
      {
      case HOST:
        if ((cpu_updated_[j] == false) && (gpu_updated_[j] == true))
        {
          syncData(j, memspace);
        }
        return &h_data_[j * n_size_];
      case DEVICE:
        if ((gpu_updated_[j] == false) && (cpu_updated_[j] == true))
        {
          syncData(j, memspace);
        }
        return &d_data_[j * n_size_];
      default:
        return nullptr;
      }
    }

    /**
     * @brief Sync out of date memory space with the updated one.
     *
     * syncData is the only function that can set data on both HOST and DEVICE
     * to the same values.
     *
     * @param[in] memspaceOut  - Memory space to sync
     *
     * @return 0 if successful, 1 otherwise.
     *
     * @warning This function can be called only when all vectors in a
     * multivector have the same update status. Otherwise, you need to sync
     * vectors in a multivector individually.
     *
     */
    int Vector::syncData(memory::MemorySpace memspaceOut)
    {
      using namespace ReSolve::memory;

      bool all_cpu_updated = cpu_updated_[0];
      bool all_gpu_updated = gpu_updated_[0];

      // Verify that all vectors in multivector have the same update status.
      for (index_type i = 1; i < k_; ++i)
      {
        if (gpu_updated_[i] != all_gpu_updated)
        {
          out::error() << "Trying to sync all multivector data on the device,"
                       << " but individual vectors were updated differently.\n"
                       << "Use syncData function for individual vectors instead!\n";
          assert(false);
          return 1;
        }
        if (cpu_updated_[i] != all_cpu_updated)
        {
          out::error() << "Trying to sync all multivector data on the host,"
                       << " but individual vectors were updated differently.\n"
                       << "Use syncData function for individual vectors instead!\n";
          assert(false);
          return 1;
        }
      }

      switch (memspaceOut)
      {
      case DEVICE: // cpu -> gpu
        if (gpu_updated_[0])
        {
          out::error() << "Trying to sync device, but device already up to date!\n";
          assert(!gpu_updated_[0]);
          return 1;
        }
        if (!cpu_updated_[0])
        {
          out::error() << "Trying to sync device with host, but host is out of date!\n";
        }
        if (d_data_ == nullptr)
        {
          // allocate first
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.copyArrayHostToDevice(d_data_, h_data_, n_size_ * k_);
        setDeviceUpdated(true);
        break;
      case HOST: // gpu -> cpu
        if (cpu_updated_[0])
        {
          out::error() << "Trying to sync host, but host already up to date!\n";
          assert(!cpu_updated_[0]);
          return 1;
        }
        if (!gpu_updated_[0])
        {
          out::error() << "Trying to sync host with device, but device is out of date!\n";
        }
        if (h_data_ == nullptr)
        {
          // allocate first
          h_data_        = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.copyArrayDeviceToHost(h_data_, d_data_, n_size_ * k_);
        setHostUpdated(true);
        break;
      default:
        return 1;
      }
      return 0;
    }

    /**
     * @brief Sync out of date memory space with the updated one.
     *
     * syncData is the only function that can set data on both HOST and DEVICE
     * to the same values.
     *
     * @param[in] memspaceOut  - Memory space to sync
     *
     * @return 0 if successful, 1 otherwise.
     *
     * @warning This function can be called only when all vectors in a
     * multivector have the same update status. Otherwise, you need to sync
     * vectors in a multivector individually.
     *
     */
    int Vector::syncData(index_type j, memory::MemorySpace memspaceOut)
    {
      using namespace ReSolve::memory;

      switch (memspaceOut)
      {
      case DEVICE: // cpu->gpu
        if (gpu_updated_[j])
        {
          out::error() << "Trying to sync device, but device already up to date!\n";
          assert(!gpu_updated_[j]);
          return 1;
        }
        if (!cpu_updated_[j])
        {
          out::error() << "Trying to sync device with host, but host is out of date!\n";
          return 1;
        }
        if (d_data_ == nullptr)
        {
          // allocate first
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.copyArrayHostToDevice(&d_data_[j * n_size_], &h_data_[j * n_size_], n_size_);
        gpu_updated_[j] = true;
        break;
      case HOST: // cuda->cpu
        if (cpu_updated_[j])
        {
          out::error() << "Trying to sync host, but host already up to date!\n";
          assert(!cpu_updated_[j]);
          return 1;
        }
        if (!gpu_updated_[j])
        {
          out::error() << "Trying to sync host with device, but device is out of date!\n";
          return 1;
        }
        if (h_data_ == nullptr)
        {
          // allocate first
          h_data_        = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.copyArrayDeviceToHost(&h_data_[j * n_size_], &d_data_[j * n_size_], n_size_);
        cpu_updated_[j] = true;
        break;
      default:
        return 1;
      }
      return 0;
    }

    /**
     * @brief Allocate vector data for HOST or DEVICE
     *
     * @param[in] memspace   - Memory space of the data to be allocated
     *
     */
    int Vector::allocate(memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        delete[] h_data_;
        h_data_        = new real_type[n_capacity_ * k_];
        owns_cpu_data_ = true;
        break;
      case DEVICE:
        mem_.deleteOnDevice(d_data_);
        mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
        owns_gpu_data_ = true;
        break;
      }
      return 0;
    }

    /**
     * @brief set vector data to zero. In case of multivectors, entire multivector is set to zero.
     *
     * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
     *
     */
    int Vector::setToZero(memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        if (h_data_ == nullptr)
        {
          h_data_        = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.setZeroArrayOnHost(h_data_, n_capacity_ * k_);
        setHostUpdated(true);
        setDeviceUpdated(false);
        break;
      case DEVICE:
        if (d_data_ == nullptr)
        {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setZeroArrayOnDevice(d_data_, n_capacity_ * k_);
        setHostUpdated(false);
        setDeviceUpdated(true);
        break;
      }
      return 0;
    }

    /**
     * @brief set the data of a single vector in a multivector to zero.
     *
     * @param[in] i          - Index of a vector in a multivector
     * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
     *
     * @pre   _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in multivector.
     */
    int Vector::setToZero(index_type j, memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        if (h_data_ == nullptr)
        {
          h_data_        = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.setZeroArrayOnHost(&h_data_[j * n_size_], n_size_);
        cpu_updated_[j] = true;
        gpu_updated_[j] = false;
        break;
      case DEVICE:
        if (d_data_ == nullptr)
        {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        // TODO: We should not need to access raw data in this class
        mem_.setZeroArrayOnDevice(&d_data_[j * n_size_], n_size_);
        cpu_updated_[j] = false;
        gpu_updated_[j] = true;
        break;
      }
      return 0;
    }

    /**
     * @brief set vector data to a given constant.
     *
     * In case of multivectors, entire multivector is set to the constant.
     *
     * @param[in] constant   - Constant (real number)
     * @param[in] memspace   - Memory space of the data to be set to constant (HOST or DEVICE)
     *
     */
    int Vector::setToConst(real_type constant, memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        if (h_data_ == nullptr)
        {
          h_data_        = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.setArrayToConstOnHost(h_data_, constant, n_size_ * k_);
        setHostUpdated(true);
        setDeviceUpdated(false);
        break;
      case DEVICE:
        if (d_data_ == nullptr)
        {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setArrayToConstOnDevice(d_data_, constant, n_size_ * k_);
        setHostUpdated(false);
        setDeviceUpdated(true);
        break;
      }
      return 0;
    }

    /**
     * @brief set the data of a single vector in a multivector to a given constant.
     *
     * @param[in] j          - Index of a vector in a multivector
     * @param[in] constant   - Constant (real number)
     * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
     *
     * @pre   _j_ < _k_ i.e,, _j_ is smaller than the total number of vectors in multivector.
     */
    int Vector::setToConst(index_type j, real_type constant, memory::MemorySpace memspace)
    {
      using namespace ReSolve::memory;
      switch (memspace)
      {
      case HOST:
        if (h_data_ == nullptr)
        {
          out::error() << "Trying to set vector host values, but the values are not allocated!" << std::endl;
          return 1;
        }
        mem_.setArrayToConstOnHost(&h_data_[n_size_ * j], constant, n_size_);
        cpu_updated_[j] = true;
        gpu_updated_[j] = false;
        break;
      case DEVICE:
        if (d_data_ == nullptr)
        {
          out::error() << "Trying to set vector device values, but the values are not allocated!" << std::endl;
          return 1;
        }
        mem_.setArrayToConstOnDevice(&d_data_[n_size_ * j], constant, n_size_);
        cpu_updated_[j] = false;
        gpu_updated_[j] = true;
        break;
      }
      return 0;
    }

    /**
     * @brief Resize vector to `new_n_size`.
     *
     * Use for vectors and multivectors that change size throughout computation.
     *
     * @note Vector needs to have capacity set to maximum expected size.
     * @warning This method is not to be used in vectors who do not own their
     * data.
     *
     * @param[in] new_n_size - New vector length
     *
     * @return 0 if successful, -1 otherwise.
     *
     * @pre `new_n_size` <= `n_capacity_`
     */
    int Vector::resize(index_type new_n_size)
    {
      assert(owns_cpu_data_ && owns_gpu_data_
             && "Cannot resize if vector is not owning the data.");

      if (new_n_size > n_capacity_)
      {
        out::error() << "Trying to resize vector to " << new_n_size
                     << " elements but memory allocated only for " << n_capacity_ << "elements."
                     << "\n";
        return 1;
      }
      else
      {
        n_size_ = new_n_size;
        return 0;
      }
    }

    /**
     * @brief copy HOST or DEVICE data of a specified vector in a multivector
     * to _dest_.
     *
     * @param[out] dest      - Pointer to the memory to which data is copied
     * @param[in] i          - Index of a vector in a multivector
     * @param[in] memspace   - Memory space (HOST or DEVICE) to copy from and to
     *
     * @return 0 if successful, -1 otherwise.
     *
     * @pre _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in
     * multivector.
     * @pre _dest_ is allocated, and the size of _dest_ is at least _n_
     * (length of a single vector in the multivector).
     * @pre _dest_ is allocated in memspaceInOut memory space.
     * @post All elements of the vector _i_ are copied to the array _dest_.
     */
    int Vector::copyDataTo(real_type*          dest,
                           index_type          i,
                           memory::MemorySpace memspaceInOut)
    {
      using namespace ReSolve::memory;
      if (i > this->k_)
      {
        return -1;
      }
      else
      {
        real_type* data = this->getData(i, memspaceInOut);
        switch (memspaceInOut)
        {
        case HOST:
          mem_.copyArrayHostToHost(dest, data, n_size_);
          break;
        case DEVICE:
          mem_.copyArrayDeviceToDevice(dest, data, n_size_);
          break;
        }
        return 0;
      }
    }

    /**
     * @brief copy HOST or DEVICE vector data to _dest_.
     *
     * In case of multivector, all data (size _k_ * _n_) is copied.
     *
     * @param[out] dest      - Pointer to the memory to which data is copied
     * @param[in] memspace   - Memory space (HOST or DEVICE) to copy from
     *
     * @return 0 if successful, -1 otherwise.
     *
     * @pre _dest_ is allocated, and the size of _dest_ is at least _k_ * _n_ .
     */
    int Vector::copyDataTo(real_type* dest, memory::MemorySpace memspaceInOut)
    {
      using namespace ReSolve::memory;
      real_type* data = this->getData(memspaceInOut);
      switch (memspaceInOut)
      {
      case HOST:
        mem_.copyArrayHostToHost(dest, data, n_size_ * k_);
        break;
      case DEVICE:
        mem_.copyArrayDeviceToDevice(dest, data, n_size_ * k_);
        break;
      }
      return 0;
    }

    //
    // Private methods
    //

    void Vector::setHostUpdated(bool is_updated)
    {
      std::fill(cpu_updated_, cpu_updated_ + k_, is_updated);
    }

    void Vector::setDeviceUpdated(bool is_updated)
    {
      std::fill(gpu_updated_, gpu_updated_ + k_, is_updated);
    }

  } // namespace vector
} // namespace ReSolve
