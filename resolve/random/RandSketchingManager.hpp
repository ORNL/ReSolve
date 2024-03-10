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
  class RandSketchingManager
  {
    private:
      using vector_type = vector::Vector;
    
    public: 
      // constructor
      RandSketchingManager()
      {
      }

      // destructor
      virtual ~RandSketchingManager()
      {
      }

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output) = 0;

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k) = 0;
      // Need to use with methods that restart
      virtual int reset() = 0;
  };
} // namespace ReSolve
