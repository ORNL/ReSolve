#pragma once
#include <string>
#include <cuda_runtime.h>
#include "Common.hpp"
#include<cstring>

namespace ReSolve 
{
  class Vector 
  {
    public:
      Vector(Int n);
      ~Vector();

      int update(Real* data, std::string memspaceIn, std::string memspaceOut);
      Real* getData(std::string memspace);

      Int getSize();

      void setDataUpdated(std::string memspace);
      void setData(Real* data, std::string memspace);
      void allocate(std::string memspace);   
    private:
      Int n_; //size
      Real* d_data_;
      Real* h_data_;
      bool gpu_updated_;
      bool cpu_updated_;
      int copyData(std::string memspaceIn, std::string memspaceOut); 
  };
}
