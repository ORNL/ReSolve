#include <cstring>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorKernels.hpp>
#include <resolve/utilities/logger/Logger.hpp>

namespace ReSolve { namespace vector {

  using out = ReSolve::io::Logger;

  /** 
   * @brief basic constructor.
   * 
   * @param[in] n - Number of elements in the vector    
   */
  Vector::Vector(index_type n):
    n_(n),
    k_(1),
    n_current_(n_),
    d_data_(nullptr),
    h_data_(nullptr),
    gpu_updated_(false),
    cpu_updated_(false),
    owns_gpu_data_(false),
    owns_cpu_data_(false)
  {
  }

  /** 
   * @brief multivector constructor.
   * 
   * @param[in] n - Number of elements in the vector    
   * @param[in] k - Number of vectors in multivector    
   */
  Vector::Vector(index_type n, index_type k)
    : n_(n),
      k_(k),
      n_current_(n_),
      d_data_(nullptr),
      h_data_(nullptr),
      gpu_updated_(false),
      cpu_updated_(false),
      owns_gpu_data_(false),
      owns_cpu_data_(false)
  {
  }

  /** 
   * @brief destructor.
   * 
   */
  Vector::~Vector()
  {
    if (owns_cpu_data_) delete [] h_data_;
    if (owns_gpu_data_) mem_.deleteOnDevice(d_data_);
  }


  /** 
   * @brief get the number of elements in a single vector.
   * 
   * @return _n_, number of elements in the vector.
   */
  index_type Vector::getSize() const
  {
    return n_;
  }

  /** 
   * @brief get the current number of elements in a single vector 
   * (use only for vectors with changing sizes, allocate for maximum expected size).
   * 
   * @return _n_current_, number of elements currently in thr vector.
   */
  index_type Vector::getCurrentSize()
  {
    return n_current_;
  }

  /** 
   * @brief get the total number of vectors in multivector.
   * 
   * @return _k_, number of vectors in multivector, or 1 if the vector is not a multivector.
   */
  index_type Vector::getNumVectors()
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
   * @brief update vector values based on another vector data.  
   * 
   * @param[in] v           - Vector, which data will be copied
   * @param[in] memspaceIn  - Memory space of the incoming data (HOST or DEVICE)  
   * @param[in] memspaceOut - Memory space the data will be copied to (HOST or DEVICE)  
   *
   * @pre   size of _v_ is equal or larger than the current vector size.
   */
  int Vector::update(Vector* v, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    real_type* data = v->getData(memspaceIn);
    return update(data, memspaceIn, memspaceOut);
  }

  /** 
   * @brief update vector data based on given input. 
   * 
   * This function allocates (if necessary) and copies the data. 
   * 
   * @param[in] data        - Data that is to be copied
   * @param[in] memspaceIn  - Memory space of the incoming data (HOST or DEVICE)  
   * @param[in] memspaceOut - Memory space the data will be copied to (HOST or DEVICE)  
   *
   * @return 0 if successful, -1 otherwise.
   */
  int Vector::update(real_type* data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::HOST))  { control = 0;}
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::DEVICE)){ control = 1;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))  { control = 2;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::DEVICE)){ control = 3;}

    if ((memspaceOut == memory::HOST) && (h_data_ == nullptr)) {
      //allocate first
      h_data_ = new real_type[n_ * k_]; 
      owns_cpu_data_ = true;
    }
    if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr)) {
      //allocate first
      mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
      owns_gpu_data_ = true;
    } 

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_data_, data, n_current_ * k_);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 2: //gpu->cpu
        mem_.copyArrayDeviceToHost(h_data_, data, n_current_ * k_);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 1: //cpu->gpu
        mem_.copyArrayHostToDevice(d_data_, data, n_current_ * k_);
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      case 3: //gpu->gpu
        mem_.copyArrayDeviceToDevice(d_data_, data, n_current_ * k_);
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
    return this->getData(0, memspace);
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
    if ((memspace == memory::HOST) && (cpu_updated_ == false) && (gpu_updated_ == true )) {
      // remember IN FIRST OUT SECOND!!!
      copyData(memory::DEVICE, memspace);  
      owns_cpu_data_ = true;
    } 

    if ((memspace == memory::DEVICE) && (gpu_updated_ == false) && (cpu_updated_ == true )) {
      copyData(memory::HOST, memspace);
      owns_gpu_data_ = true;
    }
    if (memspace == memory::HOST) {
      return &h_data_[i * n_current_];
    } else {
      if (memspace == memory::DEVICE){
        return &d_data_[i * n_current_];
      } else {
        return nullptr;
      }
    }
  }


  /** 
   * @brief copy internal vector data from HOST to DEVICE or from DEVICE to HOST 
   * 
   * @param[in] memspaceIn   - Memory space of the data to copy FROM  
   * @param[in] memspaceOut  - Memory space of the data to copy TO 
   *
   * @return 0 if successful, -1 otherwise.
   *
   */
  int Vector::copyData(memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::DEVICE)){ control = 0;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))  { control = 1;}

    if ((memspaceOut == memory::HOST) && (h_data_ == nullptr)) {
      //allocate first
      h_data_ = new real_type[n_ * k_];
      owns_cpu_data_ = true;
    }
    if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr)) {
      //allocate first
      mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
      owns_gpu_data_ = true;
    } 
    switch(control)  {
      case 0: //cpu->cuda
        mem_.copyArrayHostToDevice(d_data_, h_data_, n_current_ * k_);
        break;
      case 1: //cuda->cpu
        mem_.copyArrayDeviceToHost(h_data_, d_data_, n_current_ * k_);
        break;
      default:
        return -1;
    }
    cpu_updated_ = true;
    gpu_updated_ = true;
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
        h_data_ = new real_type[n_ * k_]; 
        owns_cpu_data_ = true;
        break;
      case DEVICE:
        mem_.deleteOnDevice(d_data_);
        mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
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
          h_data_ = new real_type[n_ * k_]; 
          owns_cpu_data_ = true;
        }
        for (int i = 0; i < n_ * k_; ++i){
          h_data_[i] = 0.0;
        }
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setZeroArrayOnDevice(d_data_, n_ * k_);
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
          h_data_ = new real_type[n_ * k_]; 
          owns_cpu_data_ = true;
        }
        for (int i = (n_current_) * j; i < n_current_ * (j + 1); ++i) {
          h_data_[i] = 0.0;
        }
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        // TODO: We should not need to access raw data in this class
        mem_.setZeroArrayOnDevice(&d_data_[j * n_current_], n_current_);
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
          h_data_ = new real_type[n_ * k_]; 
          owns_cpu_data_ = true;
        }
        for (int i = 0; i < n_ * k_; ++i){
          h_data_[i] = C;
        }
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        set_array_const(n_ * k_, C, d_data_);
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
          h_data_ = new real_type[n_ * k_]; 
          owns_cpu_data_ = true;
        }
        for (int i = j * n_current_; i < (j + 1 ) * n_current_ ; ++i) {
          h_data_[i] = C;
        }
        break;
      case DEVICE:
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        set_array_const(n_current_ * 1, C, &d_data_[n_current_ * j]);
        break;
    }
  }

  /** 
   * @brief get a pointer to HOST or DEVICE data of a specified vector in a multivector.
   * 
   * @param[in] i          - Index of a vector in a multivector
   * @param[in] memspace   - Memory space of the pointer (HOST or DEVICE)  
   *
   * @return pointer to the _i_ th vector data (HOST or DEVICE). e
   *
   * @pre   _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in multivector.
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
   * @brief Set current lenght of the vector (use for vectors and multivectors
   * that change size throughout computation). Note: vector needs to be
   * allocated using maximum expected lenght.
   * 
   * @param[in] new_n_current       - New vector lenght
   *
   * @return 0 if successful, -1 otherwise.
   *
   * @pre   _new_n_current_ <= _n_ i.e,, _new_n_current_ is smaller than the allocated vector lenght.
   */
  int Vector::setCurrentSize(int new_n_current)
  {
    if (new_n_current > n_) {
      return -1;
    } else {
      n_current_ = new_n_current;
      return 0;
    }
  }

  /** 
   * @brief copy HOST or DEVICE data of a specified vector in a multivector to _dest_. 
   * 
   * @param[out] dest      - Pointer to the memory to which data is copied
   * @param[in] i          - Index of a vector in a multivector
   * @param[in] memspace   - Memory space (HOST or DEVICE) to copy from
   *
   * @return 0 if successful, -1 otherwise.
   *
   * @pre _i_ < _k_ i.e,, _i_ is smaller than the total number of vectors in multivector.
   * @pre _dest_ is allocated, and the size of _dest_ is at least _n_ (lenght of a single vector in the multivector).
   */
  int  Vector::deepCopyVectorData(real_type* dest, index_type i, memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;
    if (i > this->k_) {
      return -1;
    } else {
      real_type* data = this->getData(i, memspaceOut);
      switch (memspaceOut) {
        case HOST:
          mem_.copyArrayHostToHost(dest, data, n_current_);
          break;
        case DEVICE:
          mem_.copyArrayDeviceToDevice(dest, data, n_current_);
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
  int  Vector::deepCopyVectorData(real_type* dest, memory::MemorySpace memspaceOut)
  {
    using namespace ReSolve::memory;
    real_type* data = this->getData(memspaceOut);
    switch (memspaceOut) {
      case HOST:
        mem_.copyArrayHostToHost(dest, data, n_current_ * k_);
        break;
      case DEVICE:
        mem_.copyArrayDeviceToDevice(dest, data, n_current_ * k_);
        break;
    }
    return 0;
  }

}} // namespace ReSolve::vector
