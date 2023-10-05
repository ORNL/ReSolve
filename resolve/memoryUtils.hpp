#pragma once

#include <cstring>
#include <iostream>

namespace ReSolve
{
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
  int copyArrayDeviceToHost(T* dst, const T* src, I n);

  template <typename I, typename T>
  int copyArrayDeviceToDevice(T* dst, const T* src, I n);

  template <typename I, typename T>
  int copyArrayHostToDevice(T* dst, const T* src, I n);

  /**
   * @brief copies a host array onto a newly allocated array on the device
   *
   * @param n - size of src array
   * @param src - array on host to be cloned
   * @param dst - array on device on which src is cloned
   *
   * @post dst is a clone of src on the device
   */
  template <typename I, typename T>
  int cloneArrayHostToDevice(I n, T** src, T** dst)
  {
    allocateArrayOnDevice(n, dst);
    copyArrayToDevice(n, *src, *dst);
    return 0;
  }

  /**
   * @brief prints array from host
   *
   * @param v - array on host
   * @param display_n - number of elements to print
   * @param label - name of array
   *
   * @pre display_n <= number of elements in v
   * @post display_n elements of v printed
   */
  template <typename I, typename T>
  void displayHostarray(T* v, 
                        I start_i,
                        I display_n, 
                        std::string label = "array")
  {
    std::cout << "\n\n" << label << ": {";
    for(int i = start_i; i < start_i + display_n - 1; i++){
      std::cout << v[i] << ", ";
    }
    std::cout << v[display_n - 1] << "}\n" << std::endl; 
  }

  /**
   * @brief prints array from device
   *
   * @param v - array on host
   * @param display_n - number of elements to print
   * @param n - number of elements in v
   * @param label - name of array
   *
   * @pre display_n <= n
   * @post display_n elements of v printed
   */
  template <typename I, typename T>
  int displayDevicearray(T* v, 
                         I n, 
                         I start_i,
                         I display_n, 
                         std::string label = "array")
  {
    T* h_v = new T[n];
    copyArrayDeviceToHost(n, v, h_v);
    displayHostArray(h_v, start_i, display_n, label);
    return 0;
  }

  /**
   * @brief clones array of size n from src to dst
   *
   * @param n - size of array
   * @param src - array to be cloned
   * @param dst - clone target
   *
   * @pre n contain an int length
   * @pre src is a valid array
   *
   * @post dst is a clone of src on device
   */
  template <typename I, typename T>
  int cloneArrayDeviceToDevice(int n, T** src, T** dst)
  {
    allocateArrayOnDevice(n, dst);
    copyArrayDeviceToDevice(n, *src, *dst);
    return 0;
  }

} // namespace ReSolve
