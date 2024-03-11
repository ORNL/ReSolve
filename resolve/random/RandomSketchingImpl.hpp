// this is the implementation virtual class
#pragma once
#include <resolve/Common.hpp>


namespace ReSolve
{
  namespace vector
  {
    class Vector;
  }
}

namespace ReSolve
{
  /** 
   * @brief Interface to random sketching implementations
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
