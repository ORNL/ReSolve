#include <cstring>
#include <cassert>
#include <resolve/vector/Vector.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve { namespace vector {

  using out = ReSolve::io::Logger;

  /** 
   * @brief Single vector constructor.
   * 
   * @param[in] n - Number of elements in the vector    
   */
  Vector::Vector(index_type n):
    n_capacity_(n),
    k_(1),
    n_size_(n)
  {
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
      n_size_(n)
  {
  }

  /**
   * @brief destructor.
   *
   */
  Vector::~Vector()
  {
    if (owns_cpu_data_ && h_data_) delete [] h_data_;
    if (owns_gpu_data_ && d_data_) mem_.deleteOnDevice(d_data_);
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
   * @brief set the vector  data variable (HOST or DEVICE) to the provided pointer.
   *
   * @param[in] data     - Pointer to data
   * @param[in] memspace - Memory space (HOST or DEVICE)
   *
   * @warning This function DOES NOT ALLOCATE any data, it only assigns the pointer.
   */
  int Vector::setData(real_type* data, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (owns_cpu_data_ && h_data_) {
          out::error() << "Trying to set vector host values, but the values already set!\n";
          out::error() << "Ignoring setData function call ...\n";
          return 1;
        }
        h_data_ = data;
        cpu_updated_ = true;
        gpu_updated_ = false;
        owns_cpu_data_ = false;
        break;
      case DEVICE:
        if (owns_gpu_data_ && d_data_) {
          out::error() << "Trying to set vector device values, but the values already set!\n";
          out::error() << "Ignoring setData function call ...\n";
          return 1;
        }
        d_data_ = data;
        gpu_updated_ = true;
        cpu_updated_ = false;
        owns_gpu_data_ = false;
        break;
    }
    return 0;
  }

  /**
   * @brief set the flag to indicate that the data (HOST or DEVICE) has been updated.
   *
   * Important because of data mirroring approach.
   *
   * @param[in] memspace - Memory space (HOST or DEVICE)
   */
  void Vector::setDataUpdated(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case DEVICE:
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
    }
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
    int control=-1;
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::HOST))  { control = 0;}
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::DEVICE)){ control = 1;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))  { control = 2;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::DEVICE)){ control = 3;}

    if ((memspaceOut == memory::HOST) && (h_data_ == nullptr)) {
      //allocate first
      h_data_ = new real_type[n_capacity_ * k_]; 
      owns_cpu_data_ = true;
    }
    if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr)) {
      //allocate first
      mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
      owns_gpu_data_ = true;
    }

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_data_, data, n_size_ * k_);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 2: //gpu->cpu
        mem_.copyArrayDeviceToHost(h_data_, data, n_size_ * k_);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 1: //cpu->gpu
        mem_.copyArrayHostToDevice(d_data_, data, n_size_ * k_);
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      case 3: //gpu->gpu
        mem_.copyArrayDeviceToDevice(d_data_, data, n_size_ * k_);
        gpu_updated_ = true;
        cpu_updated_ = false;
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
   * @return pointer to the vector data (HOST or DEVICE). In case of multivectors, vectors are stored column-wise.
   *
   * @note This function gives you access to the pointer, not to a copy.
   * If you change the values using the pointer, the vector values will change too.
   */
  real_type* Vector::getData(memory::MemorySpace memspace)
  {
    return getData(0, memspace);
  }

  /**
   * @brief get a pointer to HOST or DEVICE data of a particular vector in a multivector.
   *
   * @param[in] i         - Index of a vector in multivector
   * @param[in] memspace  - Memory space of the pointer (HOST or DEVICE)
   *
   * @return pointer to the _i_th vector data (HOST or DEVICE) within a multivector.
   *
   * @pre   _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in multivector.
   *
   * @note This function gives you access to the pointer, not to a copy.
   * If you change the values using the pointer, the vector values will change too.
   */
  real_type* Vector::getData(index_type i, memory::MemorySpace memspace)
  {
    using memory::HOST;
    using memory::DEVICE;

    switch (memspace)
    {
    case HOST:
      if ((cpu_updated_ == false) && (gpu_updated_ == true )) {
        syncData(memspace);  
      } 
      return &h_data_[i * n_size_];
    case DEVICE:
      if ((gpu_updated_ == false) && (cpu_updated_ == true )) {
        syncData(memspace);
      }
      return &d_data_[i * n_size_];
    default:
      return nullptr;
    }
  }


  /** 
   * @brief Copy internal vector data from HOST to DEVICE or from DEVICE to HOST 
   * 
   * @param[in] memspaceOut  - Memory space to sync
   *
   * @return 0 if successful, 1 otherwise.
   *
   */
  int Vector::syncData(memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;

    switch(memspaceOut)  {
      case DEVICE: // cpu->gpu
        if (gpu_updated_) {
          out::error() << "Trying to sync device, but device already up to date!\n";
          assert(!gpu_updated_);
          return 1;
        }
        if (!cpu_updated_) {
          out::error() << "Trying to sync device with host, but host is out of date!\n";
        }
        if (d_data_ == nullptr) {
          //allocate first
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        } 
        mem_.copyArrayHostToDevice(d_data_, h_data_, n_size_ * k_);
        gpu_updated_ = true;
        break;
      case HOST: //cuda->cpu
        if (cpu_updated_) {
          out::error() << "Trying to sync host, but host already up to date!\n";
          assert(!cpu_updated_);
          return 1;
        }
        if (!gpu_updated_) {
          out::error() << "Trying to sync host with device, but device is out of date!\n";
        }
        if (h_data_ == nullptr) {
          //allocate first
          h_data_ = new real_type[n_capacity_ * k_];
          owns_cpu_data_ = true;
        }
        mem_.copyArrayDeviceToHost(h_data_, d_data_, n_size_ * k_);
        cpu_updated_ = true;
        break;
      default:
        return 1;
    }
    return 0;
  }

  /**
   * @brief allocate vector data for HOST or DEVICE
   *
   * @param[in] memspace   - Memory space of the data to be allocated
   *
   */
  void Vector::allocate(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        delete [] h_data_;
        h_data_ = new real_type[n_capacity_ * k_]; 
        owns_cpu_data_ = true;
        break;
      case DEVICE:
        mem_.deleteOnDevice(d_data_);
        mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
        owns_gpu_data_ = true;
        break;
    }
  }

  /**
   * @brief set vector data to zero. In case of multivectors, entire multivector is set to zero.
   *
   * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
   *
   */
  void Vector::setToZero(memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (h_data_ == nullptr) {
          h_data_ = new real_type[n_capacity_ * k_]; 
          owns_cpu_data_ = true;
        }
        mem_.setZeroArrayOnHost(h_data_, n_capacity_ * k_);
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setZeroArrayOnDevice(d_data_, n_capacity_ * k_);
        break;
    }
  }

  /**
   * @brief set the data of a single vector in a multivector to zero.
   *
   * @param[in] i          - Index of a vector in a multivector
   * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
   *
   * @pre   _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in multivector.
   */
  void Vector::setToZero(index_type j, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (h_data_ == nullptr) {
          h_data_ = new real_type[n_capacity_ * k_]; 
          owns_cpu_data_ = true;
        }
        mem_.setZeroArrayOnHost(&h_data_[j * n_size_], n_size_);
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        // TODO: We should not need to access raw data in this class
        mem_.setZeroArrayOnDevice(&d_data_[j * n_size_], n_size_);
        break;
    }
  }

  /**
   * @brief set vector data to a given constant.
   *
   * In case of multivectors, entire multivector is set to the constant.
   *
   * @param[in] C          - Constant (real number)
   * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
   *
   */
  void Vector::setToConst(real_type C, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (h_data_ == nullptr) {
          h_data_ = new real_type[n_capacity_ * k_]; 
          owns_cpu_data_ = true;
        }
        mem_.setArrayToConstOnHost(h_data_, C, n_capacity_ * k_);
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setArrayToConstOnDevice(d_data_, C, n_capacity_ * k_);
        break;
    }
  }

  /**
   * @brief set the data of a single vector in a multivector to a given constant.
   *
   * @param[in] j          - Index of a vector in a multivector
   * @param[in] C          - Constant (real number)
   * @param[in] memspace   - Memory space of the data to be set to 0 (HOST or DEVICE)
   *
   * @pre   _j_ < _k_ i.e,, _j_ is smaller than the total number of vectors in multivector.
   */
  void Vector::setToConst(index_type j, real_type C, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        if (h_data_ == nullptr) {
          h_data_ = new real_type[n_capacity_ * k_]; 
          owns_cpu_data_ = true;
        }
        mem_.setArrayToConstOnHost(&h_data_[n_size_ * j], C, n_size_);
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_capacity_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setArrayToConstOnDevice(&d_data_[n_size_ * j], C, n_size_);
        break;
    }
  }

  /** 
   * @brief Get a pointer to HOST or DEVICE data of a specified vector in a multivector.
   * 
   * @param[in] i          - Index of a vector in a multivector
   * @param[in] memspace   - Memory space of the pointer (HOST or DEVICE)
   *
   * @return A pointer to the `i`th vector data (HOST or DEVICE).
   *
   * @pre `i` < `k_`, i.e. `i` is smaller than the total number of vectors in multivector.
   * 
   * @note This function gives you access to the pointer, not to a copy.
   * If you change the values using the pointer, the vector values will change too.
   */
  real_type* Vector::getVectorData(index_type i, memory::MemorySpace memspace)
  {
    if (this->k_ < i) {
      return nullptr;
    } else {
      return this->getData(i, memspace);
    }
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
    assert(owns_cpu_data_ || owns_gpu_data_ 
           && "Cannot resize if vector is not owning the data.");

    if (new_n_size > n_capacity_) {
      out::error() << "Trying to resize vector to " << new_n_size 
                   << " elements but memory allocated only for " << n_capacity_ << "elements." << "\n";
      return 1;
    } else {
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
  int  Vector::copyDataTo(real_type* dest,
                          index_type i,
                          memory::MemorySpace memspaceInOut)
  {
    using namespace ReSolve::memory;
    if (i > this->k_) {
      return -1;
    } else {
      real_type* data = this->getData(i, memspaceInOut);
      switch (memspaceInOut) {
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
  int  Vector::copyDataTo(real_type* dest, memory::MemorySpace memspaceInOut)
  {
    using namespace ReSolve::memory;
    real_type* data = this->getData(memspaceInOut);
    switch (memspaceInOut) {
      case HOST:
        mem_.copyArrayHostToHost(dest, data, n_size_ * k_);
        break;
      case DEVICE:
        mem_.copyArrayDeviceToDevice(dest, data, n_size_ * k_);
        break;
    }
    return 0;
  }

}} // namespace ReSolve::vector
