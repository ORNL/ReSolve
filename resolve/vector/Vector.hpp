#pragma once
#include <string>
#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve { namespace vector {
  /**
   * @brief This class implements vectors (dense arrays) and multivectors and
   * some basic utilities (get size, allocate, set data, get data, etc).
   *       
   * 
   * What you need to know:
   *  - Multivectors are stored in one array, organized column-wise. A vector
   *    is a multivector of size 1.
   *  - There is a mirroring memory approach: the class has DEVICE and HOST
   *    data pointers. If needed,  only one (or none) can be used or allocated.
   *    Unless triggered directly or by other function, the data is NOT
   *    automatically updated between HOST and DEVICE.
   *  - Constructor DOES NOT allocate memory. This has to be done separately.
   *  - You can get (and set) "raw" data easily, if needed. 
   *  - There is memory ownership utility - vector can own memory (separate
   *    flags for HOST and DEVICE) or not, depending on how it is used.
   *
   * @author Kasia Swirydowicz <kasia.swirydowicz@pnnl.gov>
   */
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
      int setData(real_type* data, memory::MemorySpace memspace);
      void allocate(memory::MemorySpace memspace);   
      void setToZero(memory::MemorySpace memspace);
      void setToZero(index_type i, memory::MemorySpace memspace); // set i-th ivector to 0
      void setToConst(real_type C, memory::MemorySpace memspace);
      void setToConst(index_type i, real_type C, memory::MemorySpace memspace); // set i-th vector to C  - needed for unit tests, Gram Schmidt tests
      int syncData(memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut); 
      int setCurrentSize(index_type new_n_current);
      real_type* getVectorData(index_type i, memory::MemorySpace memspace); // get ith vector data out of multivector   
      int deepCopyVectorData(real_type* dest, index_type i, memory::MemorySpace memspace);  
      int deepCopyVectorData(real_type* dest, memory::MemorySpace memspace);  //copy FULL multivector 
    
    private:
      index_type n_{0}; ///< size
      index_type k_{0}; ///< k_ = 1 for vectors and k_>1 for multivectors (multivectors are accessed column-wise). 
      index_type n_current_; ///< if vectors dynamically changes size, "current n_" keeps track of this. Needed for some solver implementations. 
      real_type* d_data_{nullptr}; ///< DEVICE data array
      real_type* h_data_{nullptr}; ///< HOST data array
      bool gpu_updated_; ///< DEVICE data flag (updated or not)
      bool cpu_updated_; ///< HOST data flag (updated or not)

      bool owns_gpu_data_{false}; ///< data owneship flag for DEVICE data
      bool owns_cpu_data_{false}; ///< data ownership flag for HOST data

      MemoryHandler mem_; ///< Device memory manager object
  };
}} // namespace ReSolve::vector
