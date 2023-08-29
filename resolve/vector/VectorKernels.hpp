#pragma once

namespace ReSolve { namespace vector {

/**
 * @brief Sets values of an array to a constant.
 *
 * @param[in]  n   - length of the array
 * @param[in]  val - the value the array is set to
 * @param[out] arr - a pointer to the array
 * 
 * @pre  `arr` is allocated to size `n`
 * @post `arr` elements are set to `val`
 */
void set_array_const(index_type n, real_type val, real_type* arr);

}} // namespace ReSolve::vector