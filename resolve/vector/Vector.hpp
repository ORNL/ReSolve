#pragma once
#include <string>
#include <resolve/Common.hpp>

namespace ReSolve { namespace vector {
  class Vector 
  {
    public:
      Vector(index_type n);
      ~Vector();

      int update(real_type* data, std::string memspaceIn, std::string memspaceOut);
      real_type* getData(std::string memspace);

      index_type getSize();

      void setDataUpdated(std::string memspace);
      void setData(real_type* data, std::string memspace);
      void allocate(std::string memspace);   
      void setToZero(std::string memspace);
    private:
      index_type n_; //size
      real_type* d_data_;
      real_type* h_data_;
      bool gpu_updated_;
      bool cpu_updated_;
      int copyData(std::string memspaceIn, std::string memspaceOut); 
  };
}} // namespace ReSolve::vector
