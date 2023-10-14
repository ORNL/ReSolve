/**
 * @file MemoryUtils.cu
 * 
 * This file includes MemoryUtils.tpp and specifies what functions to
 * instantiate from function templates.
 * 
 * @author Slaven Peles <peless@ornl.gov>
 */


#include <iostream>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>

#include <resolve/MemoryUtils.tpp>

namespace ReSolve
{
  template void MemoryUtils<memory::Cuda>::deviceSynchronize();
  template int MemoryUtils<memory::Cuda>::getLastDeviceError();
  template int MemoryUtils<memory::Cuda>::deleteOnDevice(void*);

  template int MemoryUtils<memory::Cuda>::allocateArrayOnDevice<index_type,  real_type>( real_type**, index_type);
  template int MemoryUtils<memory::Cuda>::allocateArrayOnDevice<index_type, index_type>(index_type**, index_type);

  template int MemoryUtils<memory::Cuda>::allocateBufferOnDevice<size_t, void>(void** v, size_t n);

  template int MemoryUtils<memory::Cuda>::setZeroArrayOnDevice<index_type, real_type>( real_type*, index_type);

  template int MemoryUtils<memory::Cuda>::copyArrayDeviceToHost<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int MemoryUtils<memory::Cuda>::copyArrayDeviceToHost<index_type, index_type>(index_type*, const index_type*, index_type);

  template int MemoryUtils<memory::Cuda>::copyArrayDeviceToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int MemoryUtils<memory::Cuda>::copyArrayDeviceToDevice<index_type, index_type>(index_type*, const index_type*, index_type);

  template int MemoryUtils<memory::Cuda>::copyArrayHostToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int MemoryUtils<memory::Cuda>::copyArrayHostToDevice<index_type, index_type>(index_type*, const index_type*, index_type);

} //namespace ReSolve
