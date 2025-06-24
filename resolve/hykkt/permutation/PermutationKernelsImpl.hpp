/**
 * @file PermutationKernelsImpl.hpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 *
 */

#pragma once

namespace ReSolve
{
  namespace hykkt
  {
    /**
     * @class PermutationKernelsImpl
     *
     * @brief Base class for different permutation kernel implementations.
     */
    class PermutationKernelsImpl
    {
    public:
      PermutationKernelsImpl()
      {
      }

      virtual ~PermutationKernelsImpl()
      {
      }

      virtual void mapIdx(int n, const int* perm, const double* old_val, double* new_val) = 0;

      /**
       * @brief Selection sorts arr1 and arr2 w/indices
       *
       * The complexity of selection sort is O(n^2) in all cases.
       * In the future the user will be given the option to choose between
       * selection sort, insertion sort, and quicksort.
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
        for (int i = 0; i < len - 1; i++)
        {
          min_ind = i;
          for (int j = i + 1; j < len; j++)
          {
            if (arr1[j] < arr1[min_ind])
            {
              min_ind = j;
            }
          }
          if (i != min_ind)
          {
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
       * @brief swaps arr1[i] with arr1[j] and arr2[i] with arr2[j]
       *
       * @param[in,out] arr1 - first array to have values swapped
       * @param[in,out] arr2 - second array to have values swapped
       * @param[in] i - index of first value to be swapped
       * @param[in] j - index of second value to be swapped
       */
      inline void swap(int* arr1, int* arr2, int i, int j)
      {
        int temp = arr1[i];
        arr1[i]  = arr1[j];
        arr1[j]  = temp;

        temp    = arr2[i];
        arr2[i] = arr2[j];
        arr2[j] = temp;
      }

      /**
       * @brief helper function for quicksort
       *
       * @param[in,out] arr1 - array to be sorted based on itself
       * @param[in,out] arr2 - array to be sorted based on other array
       * @param[in] low - lower index bound of array slice
       * @param[in] high - higher index bound of array slice
       * @return int - index of the pivot
       */
      inline int partition(int* arr1, int* arr2, int low, int high)
      {
        int pivot = arr1[high];
        int i     = (low - 1);
        for (int j = low; j <= high - 1; j++)
        {
          if (arr1[j] < pivot)
          {
            i++;
            swap(arr1, arr2, i, j);
          }
        }
        swap(arr1, arr2, i + 1, high);
        return (i + 1);
      }

      /**
       * @brief quicksorts arr1 and arr2 between indices low and high
       *
       * The complexity of quicksort is O(n log n) in the average case,
       * but O(n^2) in the worst case. For our test cases, n is small,
       * so quicksort is not a good choice, therefore we use insertion sort.
       * In the future the user will be given the option to choose between
       * selection sort, insertion sort, and quicksort.
       *
       *
       * @param[in,out] arr1 - input array to be sorted
       * @param[in,out] arr2 - array to be sorted based on other array
       * @param[in] low - lower index bound of array slice
       * @param[in] high - higher index bound of array slice
       */
      void quickSort(int* arr1, int* arr2, int low, int high)
      {
        if (low < high)
        {
          int pi = partition(arr1, arr2, low, high);
          quickSort(arr1, arr2, low, pi - 1);
          quickSort(arr1, arr2, pi + 1, high);
        }
      }

      /**
       * @brief Insertion sorts arr1 and arr2 w/indices
       * based on increasing value in arr1
       *
       * The complexity of insertion sort is O(n^2) in the worst case.
       * It is chosen here because it is simple and efficient for small n.
       * In the future the user will be given the option to choose between
       * selection sort, insertion sort, and quicksort.
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
        for (i = 1; i < n; i++)
        {
          key1 = arr1[i];
          key2 = arr2[i];

          j = i - 1;

          while (j >= 0 && arr1[j] > key1)
          {
            arr1[j + 1] = arr1[j];
            arr2[j + 1] = arr2[j];
            j           = j - 1;
          }
          arr1[j + 1] = key1;
          arr2[j + 1] = key2;
        }
      }

      /**
       * @brief Permutes the columns in a matrix represented by rows and cols
       *
       * @param[in] n - size of the matrix
       * @param[in] rows - row offsets of matrix
       * @param[in] cols - column indices of matrix
       * @param[in] rev_perm - reverse permutation
       * @param[out] perm_cols - permuted column array
       * @param[out] perm_map - corresponding indices to facilitate permuting the values
       *
       * @pre rev_perm has integers 0 to n-1 (permuted),
       * rows and cols present valid csr storage array
       *
       * @post perm_cols is now the permuted column array and perm_map stores
       * the corresponding indices to facilitate permuting the values
       */
      void makeVecMapC(int        n,
                       const int* rows,
                       const int* cols,
                       const int* rev_perm,
                       int*       perm_cols,
                       int*       perm_map)
      {
        int row_s;
        int rowlen;
        for (int i = 0; i < n; i++)
        {
          row_s  = rows[i];
          rowlen = rows[i + 1] - row_s;
          for (int j = 0; j < rowlen; j++)
          {
            perm_map[row_s + j]  = row_s + j;
            perm_cols[row_s + j] = rev_perm[cols[row_s + j]];
          }
          insertionSort(rowlen, &perm_cols[row_s], &perm_map[row_s]);
        }
      }

      /**
       * @brief Creates a reverse permutation based on a given permutation
       *
       * @param[in] n - size of the permutation
       * @param[in] perm - permutation array
       * @param[out] rev_perm - reversed permutation array
       *
       * @pre perm has integers 0 to n-1 (permuted),
       *
       * @post rev_perm now contains the reverse permutation
       */
      void reversePerm(int n, const int* perm, int* rev_perm)
      {
        for (int i = 0; i < n; i++)
        {
          rev_perm[perm[i]] = i;
        }
      }

      /**
       * @brief Permutes the rows in a matrix represented by rows and cols
       *
       * @param[in] n - size of the matrix
       * @param[in] rows - row offsets of matrix
       * @param[in] cols - column indices of matrix
       * @param[in] perm - permutation array
       * @param[out] perm_rows - row offsets of permuted matrix
       * @param[out] perm_cols - column indices of permuted matrix
       * @param[out] perm_map - corresponding indices to facilitate permuting the values
       *
       * @pre perm has integers 0 to n-1 (permuted),
       * rows and cols present valid csr storage array
       *
       * @post perm_rows and perm_cols are now the permuted rows and column arrays,
       * perm_map stores the corresponding indices to facilitate permuting the values
       */
      void makeVecMapR(int        n,
                       const int* rows,
                       const int* cols,
                       const int* perm,
                       int*       perm_rows,
                       int*       perm_cols,
                       int*       perm_map)
      {
        perm_rows[0] = 0;
        int count    = 0;
        int idx;
        int row_s;
        int rowlen;
        for (int i = 0; i < n; i++)
        {
          idx              = perm[i];
          row_s            = rows[idx];
          rowlen           = rows[idx + 1] - row_s;
          perm_rows[i + 1] = perm_rows[i] + rowlen;
          for (int j = 0; j < rowlen; j++)
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
       * @param[in] n - size of the matrix
       * @param[in] rows - row offsets of matrix
       * @param[in] cols - column indices of matrix
       * @param[in] perm - permutation array
       * @param[in] rev_perm - reverse permutation array
       * @param[out] perm_rows - row offsets of permuted matrix
       * @param[out] perm_cols - column indices of permuted matrix
       * @param[out] perm_map - corresponding indices to facilitate permuting the values
       *
       * @pre perm and rev_perm have corresponding integers 0 to n-1 (permuted),
       * rows and cols present valid csr storage array
       *
       * @post perm_rows and perm_cols are now the permuted rows and column
       * arrays, perm_map stores the corresponding indices to facilitate
       * permuting the values
       */
      void makeVecMapRC(int        n,
                        const int* rows,
                        const int* cols,
                        const int* perm,
                        const int* rev_perm,
                        int*       perm_rows,
                        int*       perm_cols,
                        int*       perm_map)
      {
        perm_rows[0] = 0;
        int count    = 0;
        int idx;
        int row_s;
        int rowlen;

        for (int i = 0; i < n; i++)
        {
          idx              = perm[i];
          row_s            = rows[idx];
          rowlen           = rows[idx + 1] - row_s;
          perm_rows[i + 1] = perm_rows[i] + rowlen;
          for (int j = 0; j < rowlen; j++)
          {
            perm_map[count + j]  = row_s + j;
            perm_cols[count + j] = rev_perm[cols[row_s + j]];
          }
          insertionSort(rowlen, &perm_cols[count], &perm_map[count]);
          count += rowlen;
        }
      }
    }; // class
  } // namespace hykkt
} // namespace ReSolve
