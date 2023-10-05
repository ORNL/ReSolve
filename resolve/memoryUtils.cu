#include <iostream>

#include <resolve/Common.hpp>
#include <resolve/memoryUtils.hpp>
#include "cuda_check_errors.hpp"

namespace ReSolve
{
  void deviceSynchronize()
  {
    cudaDeviceSynchronize();
  }
  
  int getLastDeviceError()
  {
    return static_cast<int>(cudaGetLastError());
  }
  
  /** 
   * @brief deletes variable from device
   *
   * @param v - a variable on the device
   *
   * @post v is freed from the device
   */
  int deleteOnDevice(void* v)
  {
    return checkCudaErrors(cudaFree(v));
  }

  /**
   * @brief allocates array v onto device
   *
   * @param v - pointer to the array to be allocated on the device
   * @param n - number of array elements (int, size_t)
   * 
   * @tparam T - Array element type
   * @tparam I - Array index type
   *
   * @post v is now a array with size n on the device
   */
  template <typename I, typename T>
  int allocateArrayOnDevice(T** v, I n)
  {
    return checkCudaErrors(cudaMalloc((void**) v, sizeof(T) * n));
  }
  template int allocateArrayOnDevice<index_type,  real_type>( real_type**, index_type);
  template int allocateArrayOnDevice<index_type, index_type>(index_type**, index_type);

  /**
   * @brief allocates buffer v onto device.
   * 
   * The difference from the array is that buffer size is required in bytes,
   * not number of elements.
   *
   * @param v - pointer to the buffer to be allocated on the device
   * @param n - size of the buffer in bytes
   * 
   * @tparam T - Buffer element data type type (typically void)
   * @tparam I - Buffer size type (typically size_t)
   *
   * @post v is now a buffer of n bytes
   */
  template <typename I, typename T>
  int allocateBufferOnDevice(T** v, I n)
  {
    return checkCudaErrors(cudaMalloc((void**) v, n));
  }
  template int allocateBufferOnDevice<size_t, void>(void** v, size_t n);

  /**
   * @brief Sets elements of device array v to zero
   *
   * @param v - pointer to the array to be allocated on the device
   * @param n - number of the array elements to be set to zero
   * 
   * @tparam T - Array element type
   * @tparam I - Array index type
   *
   * @post First n elements of array v are set to zero
   */
  template <typename I, typename T>
  int setZeroArrayOnDevice(T* v, I n)
  {
    return checkCudaErrors(cudaMemset(v, 0, sizeof(T) * n));
  }
  template int setZeroArrayOnDevice<index_type,  real_type>( real_type*, index_type);

  /** 
   * @brief Copies array `src` from device to the array `dst` on the host.
   *
   * @param[in]    n - size of src array
   * @param[in]  src - array on device
   * @param[out] dst - array on host
   *
   * @pre `src` is a pointer to an allocated array on the device
   * @pre `dst` is allocated to size >= n on the host
   * @post Content of `dst` is overwritten by the content of `src`
   */
  template <typename I, typename T>
  int copyArrayDeviceToHost(T* dst, const T* src, I n)
  {
    return checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
  }
  template int copyArrayDeviceToHost<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int copyArrayDeviceToHost<index_type, index_type>(index_type*, const index_type*, index_type);

  /**
   * @brief Copies array `src` to the array `dst` on the device.
   *
   * @param n - size of src array
   * @param src - array on device to be copied
   * @param dst - array on device to be copied onto
   *
   * @pre `src` is a pointer to an allocated array on the device
   * @pre `dst` is allocated to size >= n on the device
   * @post Content of `dst` is overwritten by the content of `src`
   */
  template <typename I, typename T>
  int copyArrayDeviceToDevice(T* dst, const T* src, I n)
  {
    return checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice));
  }
  template int copyArrayDeviceToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int copyArrayDeviceToDevice<index_type, index_type>(index_type*, const index_type*, index_type);

  /**
   * @brief Copies array `src` from the host to the array `dst` on the device.
   *
   * @param n - size of src array
   * @param src - array on the host to be copied
   * @param dst - array on the device to be copied onto
   *
   * @pre `src` is a pointer to an allocated array on the host
   * @pre `dst` is allocated to size >= n on the device
   * @post Content of `dst` is overwritten by the content of `src`
   */
  template <typename I, typename T>
  int copyArrayHostToDevice(T* dst, const T* src, I n)
  {
    return checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
  }
  template int copyArrayHostToDevice<index_type,  real_type>( real_type*, const  real_type*, index_type);
  template int copyArrayHostToDevice<index_type, index_type>(index_type*, const index_type*, index_type);

} //namespace ReSolve
