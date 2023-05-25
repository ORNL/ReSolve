#pragma once
#include <string>
#include <cuda_runtime.h>
#include "Common.hpp"
#include<cstring>

namespace ReSolve 
{
  class resolveVector 
  {
    public:
      resolveVector(resolveInt n);
      ~resolveVector();

      int update(resolveReal* data, std::string memspaceIn, std::string memspaceOut);
      resolveReal* getData(std::string memspace);

      resolveInt getSize();

      void setDataUpdated(std::string memspace);
      void allocate(std::string memspace);   
    private:
      resolveInt n_; //size
      resolveReal* d_data_;
      resolveReal* h_data_;
      bool gpu_updated_;
      bool cpu_updated_;
      int copyData(std::string memspaceIn, std::string memspaceOut); 
  };
}
