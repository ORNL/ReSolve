#pragma once
#include <string>
#include <resolve/Common.hpp>

namespace ReSolve { namespace vector {
  class Vector 
  {
    public:
      Vector(index_type n);
      Vector(index_type n, index_type k);
      ~Vector();

      int update(real_type* data, std::string memspaceIn, std::string memspaceOut);
      real_type* getData(std::string memspace);
      real_type* getData(index_type i, std::string memspace); // get pointer to i-th vector in multivector

      index_type getSize();
      index_type getCurrentSize();
      index_type getNumVectors();

      void setDataUpdated(std::string memspace);
      void setData(real_type* data, std::string memspace);
      void allocate(std::string memspace);   
      void setToZero(std::string memspace);
      void setToZero(index_type i, std::string memspace); // set i-th ivector to 0
      void setToConst(real_type C, std::string memspace);
      void setToConst(index_type i, real_type C, std::string memspace); // set i-th vector to C  - needed for unit tests, Gram Schmidt tests
      int copyData(std::string memspaceIn, std::string memspaceOut); 
      int setCurrentSize(index_type new_n_current);
      real_type* getVectorData(index_type i, std::string memspace); // get ith vector data out of multivector   
      int  deepCopyVectorData(real_type* dest, index_type i, std::string memspace);  
      int  deepCopyVectorData(real_type* dest, std::string memspace);  //copy FULL multivector 
    
    private:
      index_type n_; ///< size
      index_type k_; ///< k_ = 1 for vectors and k_>1 for multivectors (multivectors are accessed column-wise). 
      index_type n_current_; // if vectors dynamically change size, "current n_" keeps track of this. Needed for some solver implementations. 
      real_type* d_data_{nullptr};
      real_type* h_data_{nullptr};
      bool gpu_updated_;
      bool cpu_updated_;
  };
}} // namespace ReSolve::vector
