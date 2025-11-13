/**
 * @file cpuPermutationKernels.cpp
 * @author Shaked Regev (regevs@ornl.gov)
 * @author Adham Ibrahim (ibrahimas@ornl.gov)
 *
 */

#include "CpuPermutationKernels.hpp"

namespace ReSolve
{
  namespace hykkt
  {
    /**
     * @brief Maps the values in old_val to new_val based on perm for CPU.
     * This is the concrete implementation of the pure virtual method.
     *
     * @param[in] n       - matrix size (number of elements to permute)
     * @param[in] perm    - desired permutation array (indices for new_val)
     * @param[in] old_val - the array containing the original values
     * @param[out] new_val - the array to store the permuted values
     *
     * @pre n is a positive integer, perm is an array of 0 to n-1
     * (in some order), old_val is initialized.
     * @post new_val contains the permuted old_val, where new_val[i] = old_val[perm[i]].
     */
    void CpuPermutationKernels::mapIdx(index_type n, const index_type* perm, const real_type* old_val, real_type* new_val)
    {
      for (index_type i = 0; i < n; ++i)
      {
        new_val[i] = old_val[perm[i]];
      }
    }
  } // namespace hykkt
} // namespace ReSolve
