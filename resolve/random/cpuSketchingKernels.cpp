/**
 * @file cpuSketchingKernels.cpp
 * @author your name (you@domain.com)
 * @brief CPU implementation of random sketching kernels.
 *
 */
#include "cpuSketchingKernels.h"

#include <cmath>
#include <stdio.h>

namespace ReSolve
{
  namespace cpu
  {
    /**
     * @brief Count sketch theta function.
     *
     * @param[in]  n      - input vector size
     * @param[in]  k      - output vector size
     * @param[in]  labels - vector of non-negative ints from 0 to k-1, length n
     * @param[in]  flip   - vector of 1s and -1s, length n
     * @param[in]  input  - vector of lengths n
     * @param[out] output - vector of length k
     *
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void count_sketch_theta(index_type n,
                            index_type /* k */,
                            index_type* labels,
                            index_type* flip,
                            real_type*  input,
                            real_type*  output)
    {
      real_type val;
      for (index_type i = 0; i < n; ++i)
      {
        val = input[i];
        if (flip[i] != 1)
        {
          val *= -1.0;
        }
        output[labels[i]] += val;
      }
    }

    /**
     * @brief y = D*x
     *
     * Multiply array x by diagonal matrix D and store result in array y.
     *
     * @param[in] n  - size of arrays x, y and matrix D.
     * @param[in] D  - diagonal matrix (stored as integer array).
     * @param[in] x  - input array x
     * @param[out] y - output array y
     *
     * @pre Arrays x, y, and D are allocated to size n.
     * @pre Arrays x and D are initialized.
     *
     * @post Array y is overwritten with D*x.
     */
    void FWHT_scaleByD(index_type        n,
                       const index_type* D,
                       const real_type*  x,
                       real_type*        y)
    {

      for (index_type i = 0; i < n; ++i)
      {
        if (D[i] == 1)
        {
          y[i] = x[i];
        }
        else
        {
          y[i] = (-1.0) * x[i];
        }
      }
    }

    /**
     * @brief Permute _input_ using _perm_ and store in _output_.
     *
     * @param[in]  k      - size of input and output arrays
     * @param[in]  perm   - permutation matrix (stored as an integer array)
     * @param[in]  input  - input array
     * @param[out] output - output array
     *
     * @pre Arrays input, output, and perm are allocated to size k.
     * @pre Arrays input and perm are initialized.
     *
     * @post Array output is overwritten with permuted values of input.
     */
    void FWHT_select(index_type        k,
                     const index_type* perm,
                     const real_type*  input,
                     real_type*        output)
    {
      for (index_type i = 0; i < k; ++i)
      {
        output[i] = input[perm[i]];
      }
    }

    /**
     * @brief
     *
     * @param[in]  M      - Placeholder for GPU grid size (not used here)
     * @param[in]  log2N  -
     * @param[out] h_Data -
     */
    void FWHT(index_type /* M */,
              index_type log2N,
              real_type* h_Data)
    {
      index_type h = 1;
      index_type N = static_cast<index_type>(std::pow(2.0, log2N));
      real_type  x, y;

      while (h < N)
      {
        for (index_type i = 0; i < N; i += 2 * h)
        {
          for (index_type j = i; j < i + h; ++j)
          {
            x             = h_Data[j];
            y             = h_Data[j + h];
            h_Data[j]     = x + y;
            h_Data[j + h] = x - y;
          }
        }
        // note: in "normal" FWHT there is also a division by sqrt(2) here
        h *= 2;
      }
    }

  } // namespace cpu
} // namespace ReSolve
