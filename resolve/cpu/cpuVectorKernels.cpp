#include <resolve/Common.hpp>
#include <resolve/vector/VectorKernels.hpp>


namespace ReSolve { namespace vector {


void set_array_const(index_type n, real_type val, real_type* arr)
{
  for(index_type i = 0; i < n; ++i) {
    arr[i] = val;
  }
}
// for randomized methods

  /**
   * @brief CountSketch, CPU version 
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
                         index_type k,
                         index_type* labels,
                         index_type* flip,
                         real_type* input,
                         real_type* output)
{
  real_type val;
  for (index_type i = 0; i < n; ++i){
    val = input[i];  
    if (flip[i] != 1) {
      val *= -1.0;
    } 
    output[labels[i]] += val;
  }
}
}} // namespace ReSolve::vector
