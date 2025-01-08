#pragma once

/*
 * @brief:  maps the values in old_val to new_val based on perm
 *
 * @params: Size n of the matrix, perm - desired permutation,
 * and old_val - the array to be permuted
 *
 * @pre:   n is a positive integer, perm is an array of 0 to n-1
 * (in some order), old_val is initialized
 *
 * @post:  new_val contains the permuted old_val
*/
void cpuMapIdx(int n, int* perm, double* old_val, double* new_val);
/*
 * @brief: Selection sorts arr1 and arr2 w/indices
 * based on increasing value in arr1
 *
 * @params: Size n of the matrix,
 * arr1 - the array that determines the sorting order,
 * arr2- sorted based on arr1
 *
 * @pre: arr1 and arr2 are arrays of length n
 *
 * @post: arr1 and arr2 are sorted based on increasing values in arr1
*/
void selectionSort(int len, int* arr1, int* arr2);

inline void swap(int* arr1, int* arr2, int i, int j);
inline int partition(int* arr1, int* arr2, int low, int high);
void quickSort(int* arr1, int* arr2, int low, int high);

/*
 * @brief: Insertion sorts arr1 and arr2 w/indices
 * based on increasing value in arr1
 *
 * @params: Size n of the matrix,
 * arr1 - the array that determines the sorting order,
 * arr2- sorted based on arr1
 *
 * @pre: arr1 and arr2 are arrays of length n
 *
 * @post: arr1 and arr2 are sorted based on increasing values in arr1
*/
void insertionSort(int len, int* arr1, int* arr2);

/*
 * @brief: Permutes the columns in a matrix represented by rows and cols
 *
 * @params: Size n of the matrix, rows and cols - representing the matrix,
 * rev_perm - the permutation to be applied
 *
 * @pre: rev_perm has integers 0 to n-1 (permuted), 
 * rows and cols present valid csr storage array
 *
 * @post: perm_cols is now the permuted column array and perm_map stores
 * the corresponding indices to facilitate permuting the values
 */
void makeVecMapC(int n, 
                 int* rows, 
                 int* cols, 
                 int* rev_perm, 
                 int* perm_cols, 
                 int* perm_map);
/*
 *
 * @brief: Creates a reverse permutation based on a given permutation
 *
 * @params: Size n of the vector, perm - original permutation
 *
 * @pre: perm has integers 0 to n-1 (permuted), 
 *
 * @post: rev_perm now contains the reverse permutation
 */
void reversePerm(int n, int* perm, int* rev_perm);
/*
 * @brief: Permutes the rows in a matrix represented by rows and cols
 *
 * @params: Size n of the matrix, rows and cols - representing the matrix,
 * perm - the permutation to be applied
 *
 * @pre: perm has integers 0 to n-1 (permuted), 
 * rows and cols present valid csr storage array
 *
 * @post: perm_rows and perm_cols are now the permuted rows and column arrays,
 * perm_map stores the corresponding indices to facilitate permuting the values
*/
void makeVecMapR(int n, 
                 int* rows, 
                 int* cols, 
                 int* perm, 
                 int* perm_rows, 
                 int* perm_cols, 
                 int* perm_map);
/*
 * @brief: Permutes the rows and columns in a matrix represented by rows and cols
 * 
 * @params: Size n of the matrix, rows and cols - representing the matrix,
 * perm - the permutation to be applied on the rows, 
 * rev_perm - the permutation to be applied on the columns
 * 
 * @pre: perm and rev_perm have corresponding integers 0 to n-1 (permuted), 
 * rows and cols present valid csr storage array
 *
 * @post: perm_rows and perm_cols are now the permuted rows and column arrays,
 * perm_map stores the corresponding indices to facilitate permuting the values
*/
void makeVecMapRC(int n, 
                  int* rows, 
                  int* cols, 
                  int* perm, 
                  int* rev_perm, 
                  int* perm_rows,
                  int* perm_cols, 
                  int* perm_map);
