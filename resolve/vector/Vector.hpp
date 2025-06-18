#pragma once
#include <string>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  namespace vector
  {
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
     * @author Slaven Peles <peless@ornl.gov>
     */
    class Vector
    {
    public:
      Vector(index_type n);
      Vector(index_type n, index_type k);
      ~Vector();

      int        copyDataFrom(const real_type* data, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);
      int        copyDataFrom(Vector* v, memory::MemorySpace memspaceIn, memory::MemorySpace memspaceOut);
      real_type* getData(memory::MemorySpace memspace);
      real_type* getData(index_type i, memory::MemorySpace memspace);

      index_type getCapacity() const;
      index_type getSize() const;
      index_type getNumVectors() const;

      int setDataUpdated(memory::MemorySpace memspace);
      int setDataUpdated(index_type j, memory::MemorySpace memspace);
      int setData(real_type* data, memory::MemorySpace memspace);
      int allocate(memory::MemorySpace memspace);
      int setToZero(memory::MemorySpace memspace);
      int setToZero(index_type i, memory::MemorySpace memspace);
      int setToConst(real_type C, memory::MemorySpace memspace);
      int setToConst(index_type i, real_type C, memory::MemorySpace memspace);
      int syncData(memory::MemorySpace memspaceOut);
      int syncData(index_type j, memory::MemorySpace memspaceOut);
      int resize(index_type new_n_current);
      int copyDataTo(real_type* dest, index_type i, memory::MemorySpace memspace);
      int copyDataTo(real_type* dest, memory::MemorySpace memspace);

    private:
      void setHostUpdated(bool is_updated);
      void setDeviceUpdated(bool is_updated);

      index_type n_capacity_{0};        ///< vector capacity
      index_type k_{0};                 ///< number of vectors in multivector
      index_type n_size_{0};            ///< actual size of the vector
      real_type* d_data_{nullptr};      ///< DEVICE data array
      real_type* h_data_{nullptr};      ///< HOST data array
      bool*      gpu_updated_{nullptr}; ///< DEVICE data flags (updated or not)
      bool*      cpu_updated_{nullptr}; ///< HOST data flags (updated or not)

      bool owns_gpu_data_{true}; ///< data owneship flag for DEVICE data
      bool owns_cpu_data_{true}; ///< data ownership flag for HOST data

      MemoryHandler mem_; ///< Device memory manager object
    };
  } // namespace vector
} // namespace ReSolve
