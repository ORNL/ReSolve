/**
 * @file cpuPermutationKernels.cpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @brief Kernels for matrix and vector permutations
 * 
 */
#include "cpuPermutationKernels.hpp"

namespace ReSolve
{ 
  namespace hykkt
  {
    /**
     * @brief maps the values in old_val to new_val based on perm
     * 
     * @param[in] n - matrix size
     * @param[in] perm - desired permutation
     * @param[in] old_val - the array to be permuted
     * @param[out] new_val - the permuted array
     * 
     * @pre n is a positive integer, perm is an array of 0 to n-1
     * (in some order), old_val is initialized
     *
     * @post new_val contains the permuted old_val
     */
    void cpuMapIdx(int n, const int* perm, const double* old_val, double* new_val)
    {
        for (int i = 0; i < n; i++) {
          new_val[i] = old_val[perm[i]];
        }
    }

    /**
     * @brief Selection sorts arr1 and arr2 w/indices
     * 
     * @param[in] len  - Size n of the matrix,
     * @param[in,out] arr1 - the array that determines the sorting order, 
     * @param[in,out] arr2 - sorted based on arr1
     * 
     * @pre arr1 and arr2 are arrays of length n
     *
     * @post arr1 and arr2 are sorted based on increasing values in arr1
     */
    void selectionSort(int len, int* arr1, int* arr2)
    {
      int min_ind;
      int temp;
      for(int i = 0; i < len - 1; i++) {
        min_ind = i;
        for(int j = i + 1; j < len; j++) {
          if(arr1[j] < arr1[min_ind]) {
            min_ind = j;
          }
        }
        if(i != min_ind) {
          temp          = arr1[i];
          arr1[i]       = arr1[min_ind];
          arr1[min_ind] = temp;
          temp          = arr2[i];
          arr2[i]       = arr2[min_ind];
          arr2[min_ind] = temp;
        }
      }
    }

    /**
     * @brief 
     * 
     * @param[in,out] arr1 
     * @param[in,out] arr2 
     * @param[in] i 
     * @param[in] j 
     */
    inline void swap(int* arr1, int* arr2, int i, int j)
    {
      int temp     = arr1[i];
      arr1[i]      = arr1[j];
      arr1[j]      = temp;

      temp          = arr2[i];
      arr2[i]       = arr2[j];
      arr2[j] = temp;
    }

    /**
     * @brief 
     * 
     * @param[in,out] arr1 
     * @param[in,out] arr2 
     * @param[in] low 
     * @param[in] high 
     * @return int 
     */
    inline int partition(int* arr1, int* arr2, int low, int high)
    {
      int pivot = arr1[high];
      int i = (low - 1);
      for (int j = low; j <= high - 1; j++) {
        if (arr1[j] < pivot) {
          i++;
          swap(arr1, arr2, i, j);
        }
      }
      swap(arr1, arr2, i + 1, high);
      return (i + 1);
    }

    /**
     * @brief 
     * 
     * @param[in,out] arr1 
     * @param[in,out] arr2 
     * @param[in] low 
     * @param[in] high 
     */
    void quickSort(int* arr1, int* arr2, int low, int high)
    {
      if (low < high) {
        int pi = partition(arr1, arr2, low, high);
        quickSort(arr1, arr2, low, pi - 1);
        quickSort(arr1, arr2, pi + 1, high);
      }
    }

    /**
     * @brief Insertion sorts arr1 and arr2 w/indices
     * based on increasing value in arr1
     * 
     * @param[in] n    - Size of the matrix,
     * @param[in,out] arr1 - the array that determines the sorting order, 
     * @param[in,out] arr2 - sorted based on arr1 
     *
     * @pre arr1 and arr2 are arrays of length n
     *
     * @post arr1 and arr2 are sorted based on increasing values in arr1
     */
    void insertionSort(int n, int* arr1, int* arr2) 
    {
      int i, key1, key2, j;
      for (i = 1; i < n; i++) {
          key1 = arr1[i];
          key2 = arr2[i];

          j = i - 1;

          while (j >= 0 && arr1[j] > key1) {
              arr1[j + 1] = arr1[j];
              arr2[j + 1] = arr2[j];
              j = j - 1;
          }
          arr1[j + 1] = key1;
          arr2[j + 1] = key2;
      }
    }

    /**
     * @brief Permutes the columns in a matrix represented by rows and cols
     * 
     * @param[in] n 
     * @param[in] rows 
     * @param[in] cols 
     * @param[in] rev_perm 
     * @param[out] perm_cols 
     * @param[out] perm_map 
     *
     * @pre rev_perm has integers 0 to n-1 (permuted), 
     * rows and cols present valid csr storage array
     *
     * @post perm_cols is now the permuted column array and perm_map stores
     * the corresponding indices to facilitate permuting the values
     */
    void makeVecMapC(int n, 
                    const int* rows, 
                    const int* cols, 
                    const int* rev_perm, 
                    int* perm_cols, 
                    int* perm_map)
    {
      int row_s;
      int rowlen;
      for(int i = 0; i < n; i++) {
        row_s  = rows[i];
        rowlen = rows[i + 1] - row_s;
        for(int j = 0; j < rowlen; j++) {
          perm_map[row_s + j]  = row_s + j;
          perm_cols[row_s + j] = rev_perm[cols[row_s + j]];
        }
    // TODO: Find a way to select sorting mechanism at runtime
#if 0
        selectionSort(rowlen, &perm_cols[row_s], &perm_map[row_s]);
#else
        //quickSort(&perm_cols[row_s], &perm_map[row_s], 0, rowlen-1);
        insertionSort(rowlen, &perm_cols[row_s], &perm_map[row_s]);
#endif
      }
    }

    /**
     * @brief Creates a reverse permutation based on a given permutation
     * 
     * @param[in] n 
     * @param[in] perm 
     * @param[out] rev_perm 
     *
     * @pre perm has integers 0 to n-1 (permuted), 
     *
     * @post rev_perm now contains the reverse permutation
     */
    void reversePerm(int n, const int* perm, int* rev_perm)
    {
      for(int i = 0; i < n; i++) {
        rev_perm[perm[i]] = i;
      }
    }

    /**
     * @brief Permutes the rows in a matrix represented by rows and cols
     * 
     * @param[in] n 
     * @param[in] rows 
     * @param[in] cols 
     * @param[in] perm 
     * @param[out] perm_rows 
     * @param[out] perm_cols 
     * @param[out] perm_map 
     * 
     * @pre perm has integers 0 to n-1 (permuted), 
     * rows and cols present valid csr storage array
     *
     * @post perm_rows and perm_cols are now the permuted rows and column arrays,
     * perm_map stores the corresponding indices to facilitate permuting the values
     */
    void makeVecMapR(int n, 
                    const int* rows, 
                    const int* cols, 
                    const int* perm, 
                    int* perm_rows, 
                    int* perm_cols, 
                    int* perm_map)
    {
      perm_rows[0] = 0;
      int count    = 0;
      int idx;
      int row_s;
      int rowlen;
      for(int i = 0; i < n; i++) {
        idx              = perm[i];
        row_s            = rows[idx];
        rowlen           = rows[idx + 1] - row_s;
        perm_rows[i + 1] = perm_rows[i] + rowlen;
        for(int j = 0; j < rowlen; j++)
        {
          perm_map[count + j]  = row_s + j;
          perm_cols[count + j] = cols[row_s + j];
        }
        count += rowlen;
      }
    }

    /**
     * @brief Permutes the rows and columns in a matrix represented by rows
     * and cols
     * 
     * @param[in] n 
     * @param[in] rows 
     * @param[in] cols 
     * @param[in] perm 
     * @param[in] rev_perm 
     * @param[out] perm_rows 
     * @param[out] perm_cols 
     * @param[out] perm_map
     * 
     * @pre perm and rev_perm have corresponding integers 0 to n-1 (permuted), 
     * rows and cols present valid csr storage array
     *
     * @post perm_rows and perm_cols are now the permuted rows and column
     * arrays, perm_map stores the corresponding indices to facilitate
     * permuting the values
     */
    void makeVecMapRC(int n, 
                      const int* rows, 
                      const int* cols, 
                      const int* perm, 
                      const int* rev_perm, 
                      int* perm_rows,
                      int* perm_cols, 
                      int* perm_map)
    {
      perm_rows[0] = 0;
      int count    = 0;
      int idx;
      int row_s;
      int rowlen;

      for(int i = 0; i < n; i++) {
        idx              = perm[i];
        row_s            = rows[idx];
        rowlen           = rows[idx + 1] - row_s;
        perm_rows[i + 1] = perm_rows[i] + rowlen;
        for(int j = 0; j < rowlen; j++)
        {
          perm_map[count + j]  = row_s + j;
          perm_cols[count + j] = rev_perm[cols[row_s + j]];
        }
    // TODO: Find a way to select sorting mechanism at runtime
#if 0
        selectionSort(rowlen, &perm_cols[count], &perm_map[count]);
#else
        //quickSort(&perm_cols[count], &perm_map[count], 0, rowlen-1);
        insertionSort(rowlen, &perm_cols[count], &perm_map[count]);
#endif
        count += rowlen;
      }
    }
  } // namespace hykkt
} // namespace ReSolve