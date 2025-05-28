#pragma once
#include <cassert>
#include <resolve/utilities/logger/Logger.hpp>


namespace ReSolve
{
  namespace memory
  {
    /**
     * @brief Class containing dummy functions when there is no GPU support.
     * 
     * @author Slaven Peles <peless@ornl.gov>
     */
    struct Cpu
    {
      /**
       * @brief Dummy function to stand in when GPU support is not enabled.
       */
      static void deviceSynchronize()
      {
        // Nothing to synchronize
      }
      
      /**
       * @brief Dummy function to stand in when GPU support is not enabled.
       * 
       * @return Allways return success!
       */
      static int getLastDeviceError()
      {
        // not on device, nothing to get
        return 0;
      }
      
      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      static int deleteOnDevice(void* /* v */)
      {
        ReSolve::io::Logger::error() << "Trying to delete on a GPU device, but GPU support not available.\n";
        assert(false && "Trying to delete on a GPU device, but GPU support not available.");
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int allocateArrayOnDevice(T** /* v */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to allocate on a GPU device, but GPU support not available.\n";
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int allocateBufferOnDevice(T** /* v */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to allocate on a GPU device, but GPU support not available.\n";
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int setZeroArrayOnDevice(T* /* v */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to initialize array on a GPU device, but GPU support not available.\n";
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int setArrayToConstOnDevice(T* /* v */, T /* c */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to initialize array on a GPU device, but GPU support not available.\n";
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int copyArrayDeviceToHost(T* /* dst */, const T* /* src */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to copy from a GPU device, but GPU support not available.\n";
        return -1;
      }

      /** 
       * @brief Dummy function to notify us something is wrong.
       * 
       * This will be called only if GPU device support is not built, so
       * trying to access a device should indicate a bug in the code.
       *
       * @return Allways return failure!
       */
      template <typename I, typename T>
      static int copyArrayDeviceToDevice(T* /* dst */, const T* /* src */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to copy to a GPU device, but GPU support not available.\n";
        return -1;
      }

      template <typename I, typename T>
      static int copyArrayHostToDevice(T* /* dst */, const T* /* src */, I /* n */)
      {
        ReSolve::io::Logger::error() << "Trying to copy to a GPU device, but GPU support not available.\n";
        return -1;
      }

    }; // struct Cuda
  } // namespace memory

} //namespace ReSolve
