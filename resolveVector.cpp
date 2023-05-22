#include "resolveVector.hpp"

namespace ReSolve 
{
  resolveVector::resolveVector(resolveInt n):
    n_{n}
  {
    d_data_ = nullptr;
    h_data_ = nullptr;
    gpu_updated_ = false;
    cpu_updated_ = false;
  }

  resolveVector::~resolveVector()
  {
    if (h_data_ != nullptr) delete [] h_data_;
    if (d_data_ != nullptr) cudaFree(d_data_);
  }


  resolveInt resolveVector::getSize()
  {
    return n_;
  }

  void resolveVector::setDataUpdated(std::string memspace)
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

  int resolveVector::update(resolveReal* data, std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ control = 0;}
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 1;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 2;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ control = 3;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new resolveReal[n_]; 
    }
    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(&d_data_, (n_) * sizeof(resolveReal)); 
    } 

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(h_data_, data, (n_) * sizeof(resolveReal));
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 2: //cuda->cpu
        cudaMemcpy(h_data_, data, (n_) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        cpu_updated_ = true;
        gpu_updated_ = false;
        break;
      case 1: //cpu->cuda
        cudaMemcpy(d_data_, data, (n_) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      case 3: //cuda->cuda
        cudaMemcpy(d_data_, data, (n_) * sizeof(resolveReal), cudaMemcpyDeviceToDevice);
        gpu_updated_ = true;
        cpu_updated_ = false;
        break;
      default:
        return -1;
    }
    return 0;
  }

  resolveReal* resolveVector::getData(std::string memspace)
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

  int resolveVector::copyData(std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;
    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ control = 0;}
    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ control = 1;}

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new resolveReal[n_]; 
    }
    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(&d_data_, (n_) * sizeof(resolveReal)); 
    } 

    switch(control)  {
      case 0: //cpu->cuda
        cudaMemcpy(d_data_, h_data_, (n_) * sizeof(resolveReal), cudaMemcpyHostToDevice);
        break;
      case 1: //cuda->cpu
        cudaMemcpy(h_data_, d_data_, (n_) * sizeof(resolveReal), cudaMemcpyDeviceToHost);
        break;
      default:
        return -1;
    }
    cpu_updated_ = true;
    gpu_updated_ = true;
    return 0;
  }

}
