// this is a virtual class
#pragma once
#include <resolve/Common.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve { 
  class RandSketchingManager {
    using vector_type = vector::Vector;
    
    public: 
      // constructor
      RandSketchingManager();

      // destructor
      virtual ~RandSketchingManager();

      // Actual sketching process
      virtual int Theta(vector_type* input, vector_type* output) = 0;

      // Setup the parameters, sampling matrices, permuations, etc
      virtual int setup(index_type n, index_type k) = 0;
      virtual int reset() = 0;

      virtual index_type getVectorSize();
      virtual index_type getSketchSize();
      virtual index_type getPaddedSize();

    protected:
      index_type n_;// size of base vector
      index_type k_rand_; // size of sketched vector
      index_type N_; // padded n -- generally N_ > n_
    
      MemoryHandler mem_; ///< Device memory manager object
  };
} // namespace ReSolve
