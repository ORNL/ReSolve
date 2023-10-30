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


  index_type Vector::getSize()
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

  void Vector::setData(real_type* data, std::string memspace)
  {

    if (memspace == "cpu") {
      h_data_ = data;
      cpu_updated_ = true;
      gpu_updated_ = false;
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        d_data_ = data;
        gpu_updated_ = true;
        cpu_updated_ = false;
      } else {
        //error
      } 
    }
  }

  void Vector::setDataUpdated(std::string memspace)
  { 
    if (memspace == "cpu") {
      cpu_updated_ = true;
      gpu_updated_ = false;
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        gpu_updated_ = true;
        cpu_updated_ = false;
      } else {
        //error
      } 
    }
  }

  int Vector::update(real_type* data, std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && ((memspaceOut == "cuda") || (memspaceOut == "hip"))){ control = 1;}
    if (((memspaceIn == "cuda") || (memspaceIn == "hip")) && (memspaceOut == "cpu")){ control = 2;}
    if (((memspaceIn == "cuda") || (memspaceIn == "hip")) && ((memspaceOut == "cuda") || (memspaceOut == "hip"))){ control = 3;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[n_ * k_]; 
    }
    if (((memspaceOut == "cuda") || (memspaceOut == "hip")) && (d_data_ == nullptr)){
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

  real_type* Vector::getData(std::string memspace)
  {
    return this->getData(0, memspace);
  }

  real_type* Vector::getData(index_type i, std::string memspace)
  {
    if ((memspace == "cpu") && (cpu_updated_ == false) && (gpu_updated_ == true )) {
      copyData(memspace, "cpu");
      owns_cpu_data_ = true;
    } 

    if (((memspace == "cuda") || (memspace == "hip")) && (gpu_updated_ == false) && (cpu_updated_ == true )) {
      copyData("cpu", memspace);
      owns_gpu_data_ = true;
    }
    if (memspace == "cpu") {
      return &h_data_[i * n_current_];
    } else {
      if ((memspace == "cuda") || (memspace == "hip")){
        return &d_data_[i * n_current_];
      } else {
        return nullptr;
      }
    }
  }


  int Vector::copyData(std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == "cpu") && ((memspaceOut == "cuda") || (memspaceOut == "hip"))){ control = 0;}
    if (((memspaceIn == "cuda") || (memspaceIn == "hip")) && (memspaceOut == "cpu")){ control = 1;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[n_ * k_]; 
    }
    if (((memspaceOut == "cuda") || (memspaceOut == "hip")) && (d_data_ == nullptr)){
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

  void Vector::allocate(std::string memspace) 
  {
    if (memspace == "cpu") {
      delete [] h_data_;
      h_data_ = new real_type[n_ * k_]; 
      owns_cpu_data_ = true;
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        mem_.deleteOnDevice(d_data_);
        mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
        owns_gpu_data_ = true;
      } else {
        std::cout<<"wrong memspace " <<memspace<<" "<<std::endl;
      }
    }
  }


  void Vector::setToZero(std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[n_ * k_]; 
        owns_cpu_data_ = true;
      }
      for (int i = 0; i < n_ * k_; ++i){
        h_data_[i] = 0.0;
      }
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        mem_.setZeroArrayOnDevice(d_data_, n_ * k_);
      }
    }
  }

  void Vector::setToZero(index_type j, std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[n_ * k_]; 
        owns_cpu_data_ = true;
      }
      for (int i = (n_current_) * j; i < n_current_ * (j + 1); ++i) {
        h_data_[i] = 0.0;
      }
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        // TODO: We should not need to access raw data in this class
        mem_.setZeroArrayOnDevice(&d_data_[j * n_current_], n_current_);
      }
    }
  }

  void Vector::setToConst(real_type C, std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[n_ * k_]; 
        owns_cpu_data_ = true;
      }
      for (int i = 0; i < n_ * k_; ++i){
        h_data_[i] = C;
      }
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        set_array_const(n_ * k_, C, d_data_);
      }
    }
  }

  void Vector::setToConst(index_type j, real_type C, std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[n_ * k_]; 
        owns_cpu_data_ = true;
      }
      for (int i = j * n_current_; i < (j + 1 ) * n_current_ * k_; ++i){
        h_data_[i] = C;
      }
    } else {
      if ((memspace == "cuda") || (memspace == "hip")) {
        if (d_data_ == nullptr) {
          mem_.allocateArrayOnDevice(&d_data_, n_ * k_);
          owns_gpu_data_ = true;
        }
        set_array_const(n_current_ * 1, C, &d_data_[n_current_ * j]);
      }
    }
  }

  real_type* Vector::getVectorData(index_type i, std::string memspace)
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

  int  Vector::deepCopyVectorData(real_type* dest, index_type i, std::string memspaceOut)
  {
    if (i > this->k_) {
      return -1;
    } else {
      real_type* data = this->getData(i, memspaceOut);
      if (memspaceOut == "cpu") {
        mem_.copyArrayHostToHost(dest, data, n_current_);
      } else {
      if ((memspaceOut == "cuda") || (memspaceOut == "hip")) {
          mem_.copyArrayDeviceToDevice(dest, data, n_current_);
        } else {
          //error
        } 
      }
      return 0;
    }
  }  

  int  Vector::deepCopyVectorData(real_type* dest, std::string memspaceOut)
  {
    real_type* data = this->getData(memspaceOut);
    if (memspaceOut == "cpu") {
      mem_.copyArrayHostToHost(dest, data, n_current_ * k_);
    } else {
      if ((memspaceOut == "cuda") || (memspaceOut == "hip")) {
        mem_.copyArrayDeviceToDevice(dest, data, n_current_ * k_);
      } else {
        //error
      } 
    }
    return 0;

  }
}} // namespace ReSolve::vector
