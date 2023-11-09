#pragma once
#include <string>
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve { namespace vector {
  class Vector 
  {
    public:
      Vector(index_type n);
      Vector(index_type n, index_type k);
      ~Vector();

      int update(real_type* data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);
      int update(Vector* v, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);
      real_type* getData(memory::MemorySpace memspace);
      real_type* getData(index_type i, memory::MemorySpace memspace); // get pointer to i-th vector in multivector

      index_type getSize() const;
      index_type getCurrentSize();
      index_type getNumVectors();

      void setDataUpdated(memory::MemorySpace memspace);
      void setData(real_type* data, memory::MemorySpace memspace);
      void allocate(memory::MemorySpace memspace);   
      void setToZero(memory::MemorySpace memspace);
      void setToZero(index_type i, memory::MemorySpace memspace); // set i-th ivector to 0
      void setToConst(real_type C, memory::MemorySpace memspace);
      void setToConst(index_type i, real_type C, memory::MemorySpace memspace); // set i-th vector to C  - needed for unit tests, Gram Schmidt tests
      int copyData(memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut); 
      int setCurrentSize(index_type new_n_current);
      real_type* getVectorData(index_type i, memory::MemorySpace memspace); // get ith vector data out of multivector   
      int deepCopyVectorData(real_type* dest, index_type i, memory::MemorySpace memspace);  
      int deepCopyVectorData(real_type* dest, memory::MemorySpace memspace);  //copy FULL multivector 
    
    private:
      index_type n_; ///< size
      index_type k_; ///< k_ = 1 for vectors and k_>1 for multivectors (multivectors are accessed column-wise). 
      index_type n_current_; // if vectors dynamically change size, "current n_" keeps track of this. Needed for some solver implementations. 
      real_type* d_data_{nullptr};
      real_type* h_data_{nullptr};
      bool gpu_updated_;
      bool cpu_updated_;

      bool owns_gpu_data_{false};
      bool owns_cpu_data_{false};

      MemoryHandler mem_; ///< Device memory manager object
  };
}} // namespace ReSolve::vector
