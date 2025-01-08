#include "cpuHykktPermutationKernels.hpp"

void cpuMapIdx(int n, int* perm, double* old_val, double* new_val)
{
    for (int i = 0; i < n; i++) {
      new_val[i] = old_val[perm[i]];
    }
}

void selectionSort(int len, int* arr1, int* arr2)
{
  int min_ind;
  int temp;
  for(int i = 0; i < len - 1; i++)
  {
    min_ind = i;
    for(int j = i + 1; j < len; j++)
    {
      if(arr1[j] < arr1[min_ind])
      {
        min_ind = j;
      }
    }
    if(i != min_ind)
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

inline void swap(int* arr1, int* arr2, int i, int j) {
  int temp     = arr1[i];
  arr1[i]      = arr1[j];
  arr1[j]      = temp;

  temp          = arr2[i];
  arr2[i]       = arr2[j];
  arr2[j] = temp;
}

inline int partition(int* arr1, int* arr2, int low, int high) {
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

void quickSort(int* arr1, int* arr2, int low, int high) {
  if (low < high) {
    int pi = partition(arr1, arr2, low, high);
    quickSort(arr1, arr2, low, pi - 1);
    quickSort(arr1, arr2, pi + 1, high);
  }
}

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
          j = j - 1;
      }
      arr1[j + 1] = key1;
      arr2[j + 1] = key2;
  }
}

void makeVecMapC(int n, 
    int* rows, 
    int* cols, 
    int* rev_perm, 
    int* perm_cols, 
    int* perm_map)
{
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
    row_s  = rows[i];
    rowlen = rows[i + 1] - row_s;
    for(int j = 0; j < rowlen; j++)
    {
      perm_map[row_s + j]  = row_s + j;
      perm_cols[row_s + j] = rev_perm[cols[row_s + j]];
    }
#if 0
    selectionSort(rowlen, &perm_cols[row_s], &perm_map[row_s]);
#else
    //quickSort(&perm_cols[row_s], &perm_map[row_s], 0, rowlen-1);
    insertionSort(rowlen, &perm_cols[row_s], &perm_map[row_s]);
#endif
  }
}

void reversePerm(int n, int* perm, int* rev_perm)
{
  for(int i = 0; i < n; i++)
  {
    rev_perm[perm[i]] = i;
  }
}

void makeVecMapR(int n, 
    int* rows, 
    int* cols, 
    int* perm, 
    int* perm_rows, 
    int* perm_cols, 
    int* perm_map)
{
  perm_rows[0] = 0;
  int count    = 0;
  int idx;
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
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

void makeVecMapRC(int n, 
    int* rows, 
    int* cols, 
    int* perm, 
    int* rev_perm, 
    int* perm_rows,
    int* perm_cols, 
    int* perm_map)
{
  perm_rows[0] = 0;
  int count    = 0;
  int idx;
  int row_s;
  int rowlen;
  for(int i = 0; i < n; i++)
  {
    idx              = perm[i];
    row_s            = rows[idx];
    rowlen           = rows[idx + 1] - row_s;
    perm_rows[i + 1] = perm_rows[i] + rowlen;
    for(int j = 0; j < rowlen; j++)
    {
      perm_map[count + j]  = row_s + j;
      perm_cols[count + j] = rev_perm[cols[row_s + j]];
    }
#if 0
    selectionSort(rowlen, &perm_cols[count], &perm_map[count]);
#else
    //quickSort(&perm_cols[count], &perm_map[count], 0, rowlen-1);
    insertionSort(rowlen, &perm_cols[count], &perm_map[count]);
#endif
    count += rowlen;
  }
}
