/**
 * @file RandomSketchingImpl.hpp
 * @author Slaven Peles (peless@ornl.gov)
 * @brief Pure virtual RandomSketchingImpl class.
 *
 */
#pragma once
#include <resolve/Common.hpp>

namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
} // namespace ReSolve

namespace ReSolve
{
  /**
   * @brief Interface to random sketching implementations.
   *
   * All sketching methods inherit from this class.
   */
  class RandomSketchingImpl
  {
  public:
    RandomSketchingImpl()
    {
    }

    virtual ~RandomSketchingImpl()
    {
    }

    // Actual sketching process
    virtual int Theta(vector::Vector* input, vector::Vector* output) = 0;

    // Setup the parameters, sampling matrices, permuations, etc
    virtual int setup(index_type n, index_type k) = 0;

    // Needed for iterative methods with restarting
    virtual int reset() = 0;
  };
} // namespace ReSolve
