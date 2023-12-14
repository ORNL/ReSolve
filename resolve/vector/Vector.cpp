#include <cstring>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorKernels.hpp>

namespace ReSolve { namespace vector {

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

  Vector::~Vector()
  {
    if (owns_cpu_data_) delete [] h_data_;
    if (owns_gpu_data_) mem_.deleteOnDevice(d_data_);
  }


  index_type Vector::getSize() const
  {
    return n_;
  }

  index_type Vector::getCurrentSize()
  {
    return n_current_;
  }

  index_type Vector::getNumVectors()
  {
    return k_;
  }

  void Vector::setData(real_type* data, memory::MemorySpace memspace)
  {
    using namespace ReSolve::memory;
    switch (memspace) {
      case HOST:
        h_data_ = data;
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case DEVICE:
        d_data_ = data;
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
    }
  }

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

  int Vector::update(Vector* v, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    real_type* data = v->getData(memspaceIn);
    return update(data, memspaceIn, memspaceOut);
  }

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
    }
    if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr)) {
      //allocate first
      mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
    } 

    switch(control)  {
      case 0: //cpu->cpu
        mem_.copyArrayHostToHost(h_data_, data, n_current_ * k_);
        owns_cpu_data_ = true;
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 2: //gpu->cpu
        mem_.copyArrayDeviceToHost(h_data_, data, n_current_ * k_);
        owns_gpu_data_ = true;
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 1: //cpu->gpu
        mem_.copyArrayHostToDevice(d_data_, data, n_current_ * k_);
        owns_gpu_data_ = true;
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      case 3: //gpu->gpu
        mem_.copyArrayDeviceToDevice(d_data_, data, n_current_ * k_);
        owns_gpu_data_ = true;
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      default:
        return -1;
    }
    return 0;
  }

  real_type* Vector::getData(memory::MemorySpace memspace)
  {
    return this->getData(0, memspace);
  }

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


  int Vector::copyData(memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == memory::HOST)   && (memspaceOut == memory::DEVICE)){ control = 0;}
    if ((memspaceIn == memory::DEVICE) && (memspaceOut == memory::HOST))  { control = 1;}

    if ((memspaceOut == memory::HOST) && (h_data_ == nullptr)) {
      //allocate first
      h_data_ = new real_type[n_ * k_]; 
    }
    if ((memspaceOut == memory::DEVICE) && (d_data_ == nullptr)) {
      //allocate first
      mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
    } 
    switch(control)  {
      case 0: //cpu->cuda
        mem_.copyArrayHostToDevice(d_data_, h_data_, n_current_ * k_);
        owns_gpu_data_ = true;
        break;
      case 1: //cuda->cpu
        mem_.copyArrayDeviceToHost(h_data_, d_data_, n_current_ * k_);
        owns_cpu_data_ = true;
        break;
      default:
        return -1;
    }
    cpu_updated_ = true;
    gpu_updated_ = true;
    return 0;
  }

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

  real_type* Vector::getVectorData(index_type i, memory::MemorySpace memspace)
  {
    if (this->k_ < i){
      return nullptr;
    } else {
      return this->getData(i, memspace);
    }
  }

  int Vector::setCurrentSize(int new_n_current)
  {
    if (new_n_current > n_) {
      return -1;
    } else {
      n_current_ = new_n_current;
      return 0;
    }
  }

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
