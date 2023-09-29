#include <cstring>
#include <cuda_runtime.h>
#include <resolve/vector/VectorMulti.hpp>
#include <resolve/vector/VectorKernels.hpp>

namespace ReSolve { namespace vector {

  VectorMulti::VectorMulti(index_type n, index_type k) : VectorBase(n)
  { 
    d_data_ = nullptr;
    h_data_ = nullptr;
    k_ = k;
    gpu_updated_ = new bool[k_]; // by default, initialized to false, no need to memset
    cpu_updated_ = new bool[k_];
  }


  VectorMulti::~VectorMulti()
  {
  }

  void  VectorMulti::setCurrentVector(index_type i)
  {
    if (i > k_) {
      // warning and set to k_ - 1;
    } else {
      if (i < 0 ) {
        // warning, and set to 0
      } else {
        current_vector_ = i;
      }
    }
  }

  index_type  VectorMulti::getCurrentVector()
  {
    return current_vector_;
  }

  void VectorMulti::setAllDataUpdated(std::string memspace)
  { 
    if (memspace == "cpu") {
      for (int ii = 0; ii < k_; ++ii) {
        cpu_updated_[vector_current_] = true;
        gpu_updated_[vector_current_] = false;
      }
    } else {
      if (memspace == "cuda") { 
        for (int ii = 0; ii < k_; ++ii) {
          gpu_updated_[vector_current_] = true;
          cpu_updated_[vector_current_] = false;
        }
      } else {
        //error
      }  
    }
  }

  int updateAll(real_type* data, std::string memspaceIn, std::string memspaceOut)
  { 
    this->setAllDataUpdated(memspaceOut);
    return  vecUpdate(memspaceIn, memspaceOut, 0, n_current_ * k_, data);
  }

  void  VectorMulti::setAllToZero(std::string memspace)
  {
    vecZero(memspace, 0, n_current_ * k_);
    this->setAllDataUpdated(memspace);
  }

  void  VectorMulti::setAllToConst(std::string memspace)
  { 
    vecConst(memspace, 0, n_current_ * k_);
    this->setAllDataUpdated(memspace);
  }

  void  VectorMulti::deepCopyAllVectorData(real_type* dest, std::string memspace)
  {
    vector_current_ = 0; 
    real_type* data = this->getData(memspaceOut);
    return vecCopyOut(memspace, 0 , k_ * n_current_, dest);
  }

  void  VectorMulti::copyAllData(std::string memspaceIn, std::string memspaceOut)
  {
    for (int ii = 0; ii < k_; ++ii) {
      cpu_updated_[vector_current_] = true;
      gpu_updated_[vector_current_] = true;
    }
    return vecCopy(memspaceIn, memspaceOut, 0, k_ * n_current_);

  }
}}

