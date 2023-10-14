/**
 * @file MemoryUtils.cpp
 * 
 * This file includes MemoryUtils.tpp and specifies what functions to
 * instantiate from function templates.
 * 
 * @author Slaven Peles <peless@ornl.gov>
 */


#include <iostream>

#include <resolve/Common.hpp>
#include <resolve/MemoryUtils.hpp>
#include <resolve/cpu/CpuMemory.hpp>

#include <resolve/MemoryUtils.tpp>

namespace ReSolve
{
  template void MemoryUtils<memory::Cpu>::deviceSynchronize();
  template int  MemoryUtils<memory::Cpu>::getLastDeviceError();
  template int  MemoryUtils<memory::Cpu>::deleteOnDevice(void*);

  template int  MemoryUtils<memory::Cpu>::allocateArrayOnDevice<index_type,  real_type>( real_type**, index_type);
  template int  MemoryUtils<memory::Cpu>::allocateArrayOnDevice<index_type, index_type>(index_type**, index_type);

  template int  MemoryUtils<memory::Cpu>::allocateBufferOnDevice<size_t, void>(void** v, size_t n);

  template int  MemoryUtils<memory::Cpu>::setZeroArrayOnDevice<index_type, real_type>( real_type*, index_type);

  template int  MemoryUtils<memory::Cpu>::copyArrayDeviceToHost<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int  MemoryUtils<memory::Cpu>::copyArrayDeviceToHost<index_type, index_type>(index_type*, const index_type*, index_type);

  template int  MemoryUtils<memory::Cpu>::copyArrayDeviceToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int  MemoryUtils<memory::Cpu>::copyArrayDeviceToDevice<index_type, index_type>(index_type*, const index_type*, index_type);

  template int  MemoryUtils<memory::Cpu>::copyArrayHostToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int  MemoryUtils<memory::Cpu>::copyArrayHostToDevice<index_type, index_type>(index_type*, const index_type*, index_type);
}
