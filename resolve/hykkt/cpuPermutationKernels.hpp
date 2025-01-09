/**
 * @file cpuPermutationKernels.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @brief Prototypes of kernels for matrix and vector permutations
 * 
 */

#pragma once

namespace ReSolve
{ 
  namespace hykkt
  {
    void cpuMapIdx(int n, const int* perm, const double* old_val, double* new_val);
    void selectionSort(int len, int* arr1, int* arr2);
    inline void swap(int* arr1, int* arr2, int i, int j);
    inline int partition(int* arr1, int* arr2, int low, int high);
    void quickSort(int* arr1, int* arr2, int low, int high);
    void insertionSort(int len, int* arr1, int* arr2);
    void makeVecMapC(int n, 
                     const int* rows, 
                     const int* cols, 
                     const int* rev_perm, 
                     int* perm_cols, 
                     int* perm_map);
    /*
    *
    * @brief: 
    *
    * @params: Size n of the vector, perm - original permutation
    */
    void reversePerm(int n, const int* perm, int* rev_perm);
    /*
    * @brief: 
    *
    * @params: Size n of the matrix, rows and cols - representing the matrix,
    * perm - the permutation to be applied
    *
    */
    void makeVecMapR(int n, 
                     const int* rows, 
                     const int* cols, 
                     const int* perm, 
                     int* perm_rows, 
                     int* perm_cols, 
                     int* perm_map);
    /*
    * @brief: 
    * 
    * @params: Size n of the matrix, rows and cols - representing the matrix,
    * perm - the permutation to be applied on the rows, 
    * rev_perm - the permutation to be applied on the columns
    * 
    */
    void makeVecMapRC(int n, 
                      const int* rows, 
                      const int* cols, 
                      const int* perm, 
                      const int* rev_perm, 
                      int* perm_rows,
                      int* perm_cols, 
                      int* perm_map);
  } // namespace hykkt
} // namespace ReSolve
