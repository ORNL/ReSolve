#include <cstring>
#include <cuda_runtime.h>
#include "Vector.hpp"

namespace ReSolve 
{
  Vector::Vector(index_type n):
    n_{n}
  {
    d_data_ = nullptr;
    h_data_ = nullptr;
    gpu_updated_ = false;
    cpu_updated_ = false;
  }

  Vector::~Vector()
  {
    if (h_data_ != nullptr) delete [] h_data_;
    if (d_data_ != nullptr) cudaFree(d_data_);
  }


  index_type Vector::getSize()
  {
    return n_;
  }

  void Vector::setData(real_type* data, std::string memspace)
  {

    if (memspace == "cpu") {
      h_data_ = data;
      cpu_updated_ = true;
    } else {
      if (memspace == "cuda") { 
        d_data_ = data;
        gpu_updated_ = true;
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
      if (memspace == "cuda") { 
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
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[n_]; 
    }
    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(&d_data_, (n_) * sizeof(real_type)); 
    } 

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_data_, data, (n_) * sizeof(real_type));
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 2: //cuda->cpu
        cudaMemcpy(h_data_, data, (n_) * sizeof(real_type), cudaMemcpyDeviceToHost);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 1: //cpu->cuda
        cudaMemcpy(d_data_, data, (n_) * sizeof(real_type), cudaMemcpyHostToDevice);
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      case 3: //cuda->cuda
        cudaMemcpy(d_data_, data, (n_) * sizeof(real_type), cudaMemcpyDeviceToDevice);
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
    if ((memspace == "cpu") && (cpu_updated_ == false) && (gpu_updated_ == true )) {
      copyData("cuda", "cpu");
    } 

    if ((memspace == "cuda") && (gpu_updated_ == false) && (cpu_updated_ == true )) {
      copyData("cpu", "cuda");
    }
    if (memspace == "cpu") {
      return h_data_;
    } else {
      if (memspace == "cuda"){
        return d_data_;
      } else {
        return nullptr;
      }
    }
  }

  int Vector::copyData(std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 0;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 1;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[n_]; 
    }
    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(&d_data_, (n_) * sizeof(real_type)); 
    } 

    switch(control)  {
      case 0: //cpu->cuda
        cudaMemcpy(d_data_, h_data_, (n_) * sizeof(real_type), cudaMemcpyHostToDevice);
        break;
      case 1: //cuda->cpu
        cudaMemcpy(h_data_, d_data_, (n_) * sizeof(real_type), cudaMemcpyDeviceToHost);
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
      if (h_data_ != nullptr) {
        delete [] h_data_;
      }
      h_data_ = new real_type[n_]; 
    } else {
      if (memspace == "cuda") {
        if (d_data_ != nullptr) {
          cudaFree(d_data_);
        }
        cudaMalloc(&d_data_, (n_) * sizeof(real_type)); 
      }
    }
  }


  void Vector::setToZero(std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[n_]; 
      }
      for (int i = 0; i < n_; ++i){
        h_data_[i] = 0.0;
      }
    } else {
      if (memspace == "cuda") {
        if (d_data_ == nullptr) {
          cudaMalloc(&d_data_, (n_) * sizeof(real_type)); 
        }
        cudaMemset(d_data_, 0.0, n_ * sizeof(real_type));
      }
    }
  }
}
