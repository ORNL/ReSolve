#pragma once

#include <resolve/resolve_defs.hpp>
#include <cstdlib>

namespace ReSolve
{
  namespace memory
  {
    enum MemorySpace{HOST = 0, DEVICE};
    enum MemoryDirection{HOST_TO_HOST = 0, HOST_TO_DEVICE, DEVICE_TO_HOST, DEVICE_TO_DEVICE};
    enum DeviceType{NONE = 0, CUDADEVICE, HIPDEVICE};
  }
}

namespace ReSolve
{
  /**
   * @class MemoryUtils
   * 
   * @brief Provides basic memory allocation, free and copy functions.
   * 
   * This class provedes abstractions for memory management functiosn for
   * different GPU programming models.
   * 
   * @tparam Policy - Memory management policy (vendor specific)
   * 
   * @author Slaven Peles <peless@ornl.gov>
   */
  template <class Policy>
  class MemoryUtils
  {
    public:
      MemoryUtils()  = default;
      ~MemoryUtils() = default;

      void deviceSynchronize();
      int getLastDeviceError();
      int deleteOnDevice(void* v);
      
      template <typename I, typename T>
      int allocateArrayOnDevice(T** v, I n);
      
      template <typename I, typename T>
      int allocateBufferOnDevice(T** v, I n);
      
      template <typename I, typename T>
      int setZeroArrayOnDevice(T* v, I n);
      
      template <typename I, typename T>
      int setArrayToConstOnDevice(T* v, T c, I n);
      
      template <typename I, typename T>
      int copyArrayDeviceToHost(T* dst, const T* src, I n);
      
      template <typename I, typename T>
      int copyArrayDeviceToDevice(T* dst, const T* src, I n);
      
      template <typename I, typename T>
      int copyArrayHostToDevice(T* dst, const T* src, I n);

      /// 
      /// Methods implemented here are always needed
      ///

      template <typename I, typename T>
      int copyArrayHostToHost(T* dst, const T* src, I n)
      {
        std::size_t arraysize = static_cast<std::size_t>(n) * sizeof(T);
        memcpy(dst, src, arraysize);
        return 0;
      }

      template <typename I, typename T>
      int setZeroArrayOnHost(T* v, I n)
      {
        std::size_t arraysize = static_cast<std::size_t>(n) * sizeof(T);
        memset(v, 0, arraysize);
        return 0;
      }

      template <typename I, typename T>
      int setArrayToConstOnHost(T* v, T c, I n)
      {
        for (I i = 0; i < n; ++i) {
          v[i] = c;
        }
        return 0;
      }
  };

} // namespace ReSolve

#ifdef RESOLVE_USE_GPU

// Check if GPU support is enabled in Re::Solve and set appropriate device memory manager.
#if defined RESOLVE_USE_CUDA
#include <resolve/cuda/CudaMemory.hpp>
using MemoryHandler = ReSolve::MemoryUtils<ReSolve::memory::Cuda>;
#elif defined RESOLVE_USE_HIP
#include <resolve/hip/HipMemory.hpp>
using MemoryHandler = ReSolve::MemoryUtils<ReSolve::memory::Hip>;
#else
#error Unrecognized device, probably bug in CMake configuration
#endif

#else

// If no GPU support is present, set device memory manager to a dummy object.
#include <resolve/cpu/CpuMemory.hpp>
using MemoryHandler = ReSolve::MemoryUtils<ReSolve::memory::Cpu>;

#endif

