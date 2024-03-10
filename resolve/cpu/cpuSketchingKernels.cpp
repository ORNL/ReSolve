#include <cmath>
#include <stdio.h>
#include "cpuSketchingKernels.h"

//
// Kernel wrappers to mimic gpu behavior and make implementation consistent
// @author Kasia Swirydowicz
//

namespace ReSolve {
  // namespace cpu {
    /**
     * @brief CountSketch Theta, CPU version 
     * 
     * @param[in]  n      - input vector size
     * @param[in]  k      - output vector size
     * @param[in]  labels - vector of non-negative ints from 0 to k-1, length n
     * @param[in]  flip   - vector of 1s and -1s, length n
     * @param[in]  input  - vector of lenghts n
     * @param[out] output - vector of lenght k
     * 
     * @todo Decide how to allow user to configure grid and block sizes.
     */
    void  count_sketch_theta(index_type n,
                            index_type /* k */,
                            index_type* labels,
                            index_type* flip,
                            real_type* input,
                            real_type* output)
    {
      real_type val;
      for (index_type i = 0; i < n; ++i) {
        val = input[i];  
        if (flip[i] != 1) {
          val *= -1.0;
        } 
        output[labels[i]] += val;
      }
    }

    void FWHT_scaleByD(index_type n,
                      const index_type* D,
                      const real_type* x,
                      real_type* y)
    {

      for (index_type i = 0; i < n; ++i) {
        if (D[i] == 1) {
          y[i] = x[i];
        } else {
          y[i] = (-1.0) * x[i];
        }
      }  
    }
  
    void FWHT_select(index_type k,
                    const index_type* perm,
                    const real_type* input,
                    real_type* output)
    {

      for (index_type i = 0; i < k; ++i) {
        output[i] = input[perm[i]];
      } 
    }

    void FWHT(index_type M, 
              index_type log2N, 
              real_type* h_Data) 
    {
      index_type h = 1;
      index_type N =  static_cast<index_type>(std::pow(2.0,log2N));
      real_type x, y;
      
      while (h < N) {
        for (index_type i = 0; i < N; i += 2*h) {
          for (index_type j = i; j < i + h; ++j) {
            x = h_Data[j];
            y = h_Data[j + h];
            h_Data[j] = x + y;
            h_Data[j + h] = x - y;
          } 
        }
        // note: in "normal" FWHT there is also a division by sqrt(2) here     
        h *= 2;
      } 
    }

  // } // namespace cpu
} // namespace ReSolve

