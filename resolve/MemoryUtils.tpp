/**
 * @file MemoryUtils.tpp
 *
 * Contains implementation of memory utility functions wrappers.
 * All it does it calls vendor specific functions frm an abstract interface.
 *
 * @author Slaven Peles <peless@ornl.gov>
 */

#pragma once

namespace ReSolve
{
  template <class Policy>
  void MemoryUtils<Policy>::deviceSynchronize()
  {
    Policy::deviceSynchronize();
  }

  template <class Policy>
  int MemoryUtils<Policy>::getLastDeviceError()
  {
    return Policy::getLastDeviceError();
  }

  template <class Policy>
  int MemoryUtils<Policy>::deleteOnDevice(void* v)
  {
    return Policy::deleteOnDevice(v);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::allocateArrayOnDevice(T** v, I n)
  {
    return Policy::template allocateArrayOnDevice<I, T>(v, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::allocateBufferOnDevice(T** v, I n)
  {
    return Policy::template allocateBufferOnDevice<I, T>(v, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::setZeroArrayOnDevice(T* v, I n)
  {
    return Policy::template setZeroArrayOnDevice<I, T>(v, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::setArrayToConstOnDevice(T* v, T c, I n)
  {
    return Policy::template setArrayToConstOnDevice<I, T>(v, c, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::copyArrayDeviceToHost(T* dst, const T* src, I n)
  {
    return Policy::template copyArrayDeviceToHost<I, T>(dst, src, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::copyArrayDeviceToDevice(T* dst, const T* src, I n)
  {
    return Policy::template copyArrayDeviceToDevice<I, T>(dst, src, n);
  }

  template <class Policy>
  template <typename I, typename T>
  int MemoryUtils<Policy>::copyArrayHostToDevice(T* dst, const T* src, I n)
  {
    return Policy::template copyArrayHostToDevice<I, T>(dst, src, n);
  }

} // namespace ReSolve
