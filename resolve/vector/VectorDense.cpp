#include <cstring>
#include <cuda_runtime.h>
#include <resolve/vector/VectorDense.hpp>
#include <resolve/vector/VectorKernels.hpp>

namespace ReSolve { namespace vector {

  VectorDense::VectorDense(index_type n) : VectorBase(n)
  {
    d_data_ = nullptr;
    h_data_ = nullptr;
    gpu_updated_ = new bool[1];
    cpu_updated_ = new bool[1];
    gpu_updated_[0] = false;
    cpu_updated_[0] = false;
  }

  VectorDense::~VectorDense()
  {
    if (h_data_ != nullptr) delete [] h_data_;
    if (d_data_ != nullptr) cudaFree(d_data_);
    
    delete [] cpu_updated_;
    delete [] gpu_updated_;
  }

  void VectorDense::setDataUpdated(std::string memspace)
  { 
    if (memspace == "cpu") {
      cpu_updated_[vector_current_] = true;
      gpu_updated_[vector_current_] = false;
    } else {
      if (memspace == "cuda") { 
        gpu_updated_[vector_current_] = true;
        cpu_updated_[vector_current_] = false;
      } else {
        //error
      } 
    }
  }

  // set sets the pointer. It always sets it AT LOCATION ZERO. DONT PLAY DIRTY TRICKS or you would generate impossible-to-debug segfaults.
  void VectorDense::setData(real_type* data, std::string memspace)
  {
    if (memspace == "cpu") {
      h_data_ = data;
      cpu_updated_[0] = true;
    } else {
      if (memspace == "cuda") { 
        d_data_ = data;
        gpu_updated_[0] = true;
      } else {
        //error
      } 
    }
  }

  // update copies so start & copy_size matter
  int VectorDense::update(real_type* data, std::string memspaceIn, std::string memspaceOut)
  {
    int control=-1;

    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ 
      control = 0;
    }

    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ 
      control = 1;
    }

    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ 
      control = 2;
    }

    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ 
      control = 3;
    }

    switch(control)  {
      case 0: //cpu->cpu
        cpu_updated_[vector_current_] = true;
        gpu_updated_[vector_current_] = false;
        break;
      case 2: //cuda->cpu
        cpu_updated_[vector_current_] = true;
        gpu_updated_[vector_current_] = false;
        break;
      case 1: //cpu->cuda
        gpu_updated_[vector_current_] = true;
        cpu_updated_[vector_current_] = false;
        break;
      case 3: //cuda->cuda
        gpu_updated_[vector_current_] = true;
        cpu_updated_[vector_current_] = false;
        break;
      default:
        return -1;
    }

    return  vecUpdate(memspaceIn, memspaceOut, vector_current_ * n_current_, n_current_, data);

  }

  // here also it matters where it starts
  real_type* VectorDense::getData(std::string memspace)
  {
    if ((memspace == "cpu") && (cpu_updated_ == false) && (gpu_updated_ == true )) {
      vecCopy("cuda", "cpu", n_current_ * vector_current_, n_currant_ );
    } 

    if ((memspace == "cuda") && (gpu_updated_ == false) && (cpu_updated_ == true )) {
      vecCopy("cpu", "cuda", n_current_ * vector_current_, n_currant_ );
    }

    return vecGet(memspace, n_current_ * vec_current_);
  }

  // here also size and start matter
  int VectorDense::copyData(std::string memspaceIn, std::string memspaceOut)
  {
    cpu_updated_[vector_current_] = true;
    gpu_updated_[vector_current_] = true;
    return vecCopy(memspaceIn, memspaceOut, n_current_ * vector_current_, n_current_);
  }

  // always allocate FULL DATA
  void VectorDense::allocate(std::string memspace) 
  {
    if (memspace == "cpu") {
      if (h_data_ != nullptr) {
        delete [] h_data_;
      }
      h_data_ = new real_type[size_alloc_]; 
    } else {
      if (memspace == "cuda") {
        if (d_data_ != nullptr) {
          cudaFree(d_data_);
        }
        cudaMalloc(&d_data_, (size_alloc_) * sizeof(real_type)); 
      }
    }
  }

  // here start and size MATTER
  void VectorDense::setToZero(std::string memspace) 
  {
    vecZero(memspace, n_current_ * vector_current_, n_current_);
    setDataUpdated(memspace);
  }


  void VectorDense::setToConst(real_type C, std::string memspace) 
  { 
    vecConst(memspace, n_current_ * vector_current_, n_current_);
    setDataUpdated(memspace);
  }

  // here start/size matter
  int  VectorDense::deepCopyVectorData(real_type* dest, std::string memspaceOut)
  {
    real_type* data = this->getData(memspaceOut);
    return vecCopyOut(memspace, n_current_ * vector_current_, n_current_, dest);
  }

  int VectorDense::vecCopy(std::string memspaceIn, std::string memspaceOut, index_type start, index_type size)
  {

    int control=-1;

    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ 
      control = 0;
    }

    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ 
      control = 1;
    }

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[size_alloc_]; 
    }

    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(&d_data_, (size_alloc_) * sizeof(real_type)); 
    } 

    switch(control)  {
      case 0: //cpu->cuda
        cudaMemcpy(&d_data_[start], &h_data_[start], (size) * sizeof(real_type), cudaMemcpyHostToDevice);
        break;
      case 1: //cuda->cpu
        cudaMemcpy(&h_data_[start], &d_data_[start], (size) * sizeof(real_type), cudaMemcpyDeviceToHost);
        break;
      default:
        return -1;
    }
    return 0;
  }

  int VectorDense::vecCopyOut(std::string memspace, index_type start, index_type size, real_type* dataOut)
  { 
    real_type* data = this->vecGet(memspace, start, size);
    if (memspaceOut == "cpu") {
      std::memcpy(dataOutt, data, size * sizeof(real_type));
    } else {
      if (memspaceOut == "cuda") { 
        cudaMemcpy(dest, data, size * sizeof(real_type), cudaMemcpyDeviceToDevice);
      } else {
        //error
      } 
    }
    return 0;
  }

  int VectorDense::vecUpdate(std::string memspaceIn, std::string memspaceOut, index_type start, index_type size, real_type* dataIn)
  {

    int control=-1;

    if ((memspaceIn == "cpu") && (memspaceOut == "cpu")){ 
      control = 0;
    }

    if ((memspaceIn == "cpu") && (memspaceOut == "cuda")){ 
      control = 1;
    }

    if ((memspaceIn == "cuda") && (memspaceOut == "cpu")){ 
      control = 2;
    }

    if ((memspaceIn == "cuda") && (memspaceOut == "cuda")){ 
      control = 3;
    }

    if ((memspaceOut == "cpu") && (h_data_ == nullptr)){
      //allocate first
      h_data_ = new real_type[size_alloc_]; 
    }
    if ((memspaceOut == "cuda") && (d_data_ == nullptr)){
      //allocate first
      cudaMalloc(d_data_, (size_alloc_) * sizeof(real_type)); 
    } 

    switch(control)  {
      case 0: //cpu->cpu
        std::memcpy(&h_data_[start], dataIn, size * sizeof(real_type));
        break;
      case 2: //cuda->cpu
        cudaMemcpy(&h_data_[start], dataIn, size * sizeof(real_type), cudaMemcpyDeviceToHost);
        break;
      case 1: //cpu->cuda
        cudaMemcpy(&d_data_[start], dataIn, size * sizeof(real_type), cudaMemcpyHostToDevice);
        break;
      case 3: //cuda->cuda
        cudaMemcpy(&d_data_[start], dataIn, size * sizeof(real_type), cudaMemcpyDeviceToDevice);
        break;
      default:
        return -1;
    }
    return 0;
  }

  void VectorDense::vecZero(std::string memespace, index_type start, index_type size)
  {

    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[size_alloc_]; 
      }
      for (int i = start; i < start + size; ++i){
        h_data_[i] = 0.0;
      }
    } else {
      if (memspace == "cuda") {
        if (d_data_ == nullptr) {
          cudaMalloc(&d_data_, (size_alloc_) * sizeof(real_type)); 
        }
        cudaMemset(&d_data_[start], 0.0, size * sizeof(real_type));
      }
    }
  }

  void VectorDense::vecConst(std::string memespace, index_type start, index_type size)
  { 
    if (memspace == "cpu") {
      if (h_data_ == nullptr) {
        h_data_ = new real_type[size_alloc_]; 
      }
      for (int i = start; i < start + size; ++i){
        h_data_[i] = C;
      }
    } else {
      if (memspace == "cuda") {
        if (d_data_ == nullptr) {
          cudaMalloc(d_data_, (size_alloc_) * sizeof(real_type)); 
        }
        set_array_const(size, C, &d_data_[start]);
      }
    }
  }

  real_type* VectorDense::vecGet(std::string memspace, index_type start)
  {  
    if (memspace == "cpu") {
      return &h_data_[start];
    } else {
      if (memspace == "cuda"){
        return &d_data_[start];
      } else {
        return nullptr;
      }
    }
  }
}} // namespace ReSolve::vector
